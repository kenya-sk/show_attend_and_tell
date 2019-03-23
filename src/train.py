import numpy as np
import pandas as pd
import os
import gc
import time
import pickle
import logging
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import  pack_padded_sequence

from utils import Vocabulary, AverageMeter, to_variable, set_transform, decode_caption, clip_gradient, adjust_learning_rate
from data_loader import ImageDataset, get_image_loader
from model import EncoderResNet, Decoder
from bleu import bleu

logger = logging.getLogger(__name__)
log_path = "../logs/train.log"
logging.basicConfig(filename=log_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def train(encoder, decoder, encoder_optimizer, decoder_optimizer, train_loader, criterion, epoch):
    """ 
    The model trained based on the parameters given by the arguments.
    """

    # training mode
    encoder.train()
    decoder.train()

    # track each time and loss
    train_data_time = AverageMeter()
    train_batch_time = AverageMeter()
    train_loss = AverageMeter()
    start = time.time()

    for images, captions, lengths in tqdm(train_loader, desc="Train Data"):
        train_data_time.update(time.time() - start)

        # initialize gradient of each model
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # convert torch variable and move to device
        imgs = to_variable(images)
        caps = to_variable(captions)
        lengths = to_variable(lengths)

        # extract feature
        features = encoder(imgs)
        features = features.view(features.size(0), args.vis_dim, args.vis_num).transpose(1,2)

        # decode caption
        predicts, alphas = decoder(features.float(), caps[:, :-1], torch.Tensor([(l - 1) for l in lengths]))
        predicts = pack_padded_sequence(predicts, [l-1 for l in lengths], batch_first=True)[0]
        caps = pack_padded_sequence(caps[:, 1:], [l-1 for l in lengths], batch_first=True)[0]

        # calculate each loss of train
        loss = criterion(predicts, caps)
        loss += float(args.alpha_c* ((1 - alphas.sum(dim=1))**2).mean())
        loss.backward(retain_graph=False)

        # clip gradient
        if args.grad_clip is not None:
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)
            clip_gradient(decoder_optimizer, args.grad_clip)

        # update each weight
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        decoder_optimizer.step()
        
        train_loss.update(float(loss))
        train_batch_time.update(time.time() - start)

        start = time.time()

        # delete temporary data
        del imgs, caps, lengths, features,predicts, alphas, loss
        gc.collect()
    
    # logging status
    logger.debug("\n*********************** TRAIN ***********************\n"
                    "EPOCH          : [{0}]\n"
                    "BATCH TIME     : {batch_time.val:.3f} ({batch_time.avg:.3f})\n"
                    "DATA LOAD TIME : {data_time.val:.3f} ({data_time.avg:.3f})\n"
                    "LOSS           : {loss.sum:.4f}\n"
                    "*****************************************************\n".format(
                    epoch,
                    batch_time=train_batch_time,
                    data_time=train_data_time,
                    loss=train_loss))


def validation(encoder, decoder, val_loader, criterion, epoch, vocab, beam_size):
    """
    The validation data is predicted using the trained model.
    retrun: DataFrame
    """

    # validation mode
    encoder.eval()
    decoder.eval()

    # store prediction and answer caption
    pred_dctlst = {"pred":[], "ans":[]}

    for images, captions, _ in tqdm(val_loader, desc="Val Data"):
        # convert torch variable and move to device
        imgs = to_variable(images)
        caps = to_variable(captions)
        
        # extract feature
        features = encoder(imgs)
        features = features.view(features.size(0), args.vis_dim, args.vis_num).transpose(1,2)
        
        # predict caption by beam search
        predicts, _ = decoder.module.beam_search_captioning(features, vocab, beam_size)

        # store result
        pred_dctlst["pred"].append(predicts)
        pred_dctlst["ans"].append(np.squeeze(captions)[1:]) # remove start

        # delete temporary data
        del imgs, features, predicts, caps
        gc.collect()

    assert len(pred_dctlst["pred"]) == len(pred_dctlst["ans"])

    return pd.DataFrame(pred_dctlst)


def main(args):
    """
    Training and validation the model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("DEVICE: {}".format(device))

    # load vocabulary
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    # encoder model setting
    encoder = EncoderResNet()
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                        lr=args.encoder_lr) if args.fine_tune_encoder else None

    # decoder model setting
    decoder = Decoder(vis_dim=args.vis_dim,
                    vis_num=args.vis_num, 
                    embed_dim=args.embed_dim,
                    hidden_dim=args.hidden_dim, 
                    vocab_size=args.vocab_size, 
                    num_layers=args.num_layers,
                    dropout_ratio=args.dropout_ratio)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr)

    # move to GPU
    encoder = nn.DataParallel(encoder).to(device)
    decoder = nn.DataParallel(decoder).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # data loader
    transform = set_transform(args.resize, args.crop_size, horizontal_flip=True, normalize=True)
    train_img_dirc = os.path.join(args.root_img_dirc, "train2014")
    train_loader = get_image_loader(train_img_dirc, args.train_data_path, vocab, transform, args.batch_size, args.shuffle, args.num_workers)
    val_img_dirc = os.path.join(args.root_img_dirc, "val2014")
    val_loader = get_image_loader(val_img_dirc, args.val_data_path, vocab, transform, 1, args.shuffle, args.num_workers)

    # initialization
    best_bleu_score = -100
    not_improved_cnt = 0

    for epoch in range(1, args.num_epochs):
        # training
        train(encoder, decoder, encoder_optimizer, decoder_optimizer, train_loader, criterion, epoch)
        
        # validation
        pred_df = validation(encoder, decoder, val_loader, criterion, epoch, vocab, args.beam_size)

        # calculate BLEU-4 score
        pred_cap_lst = decode_caption(pred_df["pred"], vocab.idx2word)
        ans_cap_lst = decode_caption(pred_df["ans"], vocab.idx2word)
        bleu_score = bleu(pred_cap_lst, ans_cap_lst, mode="4-gram")

        # early stopping
        if bleu_score < best_bleu_score:
            not_improved_cnt += 1
        else:
            # learning is going well
            best_bleu_score = bleu_score
            not_improved_cnt = 0

            # save best params model
            torch.save(encoder.state_dict(), args.save_encoder_path)
            torch.save(decoder.state_dict(), args.save_decoder_path)

        # logging status
        logger.debug("\n************************ VAL ************************\n"
                     "EPOCH          : [{0}/{1}]\n"
                     "BLEU-4         : {2}\n"
                     "EARLY STOPPING : [{3}/{4}]\n"
                     "*****************************************************\n".format(
                        epoch, args.num_epochs, bleu_score,not_improved_cnt, args.stop_count))

        if not_improved_cnt == args.stop_count:
            logger.debug("Early Stopping")
            break

        # decay learning rate if there is no improvement for 10 consecutive epochs
        if not_improved_cnt%10 == 0:
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
            adjust_learning_rate(decoder_optimizer, 0.8)


def make_parse():
    parser = argparse.ArgumentParser(
        prog="train.py",
        usage="training decoder model (default encoder model is ResNet)",
        description="description",
        epilog="end",
        add_help=True
    )

    # data Argument
    parser.add_argument("--root_img_dirc", type=str, default="../images/")
    parser.add_argument("--train_data_path", type=str, default="../data/label/train.csv")
    parser.add_argument("--val_data_path", type=str, default="../data/label/val.csv")
    parser.add_argument("--vocab_path", type=str, default="../data/vocab/vocab.pkl")
    parser.add_argument("--save_encoder_path", type=str, default="../data/model/encoder.pth")
    parser.add_argument("--save_decoder_path", type=str, default="../data/model/decoder.pth")

    # params Argument
    parser.add_argument("--fine_tune_encoder", type=bool, default=True)
    parser.add_argument("--resize", type=tuple, default=(256, 256))
    parser.add_argument("--crop_size", type=tuple, default=(224, 224))
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vis_dim", type=int, default=2048)
    parser.add_argument("--vis_num", type=int, default=196)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--alpha_c", type=int, default=1)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--decoder_lr", type=float, default=4e-4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stop_count", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--beam_size", type=int, default=3)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = make_parse()
    logger.debug("Running with args: {0}".format(args))
    main(args)