import numpy as np
import pandas as pd
import argparse
from PIL import Image
import os
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import EncoderResNet, Decoder
from utils import  to_variable, Vocabulary, set_transform, decode_caption


def save_prediciton(names_lst, captions_lst, save_dirc):
    # get pred and test dataframe
    pred_dct_lst = {"pred": [], "ans": []}
    for i in range(len(names_lst)):
        pred_dct_lst["ans"].append(names_lst[i])
        pred_dct_lst["pred"].append(captions_lst[i])

    save_path = os.path.join(save_dirc, "prediction.csv")
    pd.DataFrame(pred_dct_lst).to_csv(save_path, index=False)


def model_setting(args):
    # Encoder
    encoder_model = EncoderResNet()
    encoder_model = nn.DataParallel(encoder_model)
    encoder_model.load_state_dict(torch.load(args.encoder_model_path))

    # Decoder
    decoder_model = Decoder(vis_dim=args.vis_dim,
                            vis_num=args.vis_num,
                            embed_dim=args.embed_dim,
                            hidden_dim=args.hidden_dim,
                            vocab_size=args.vocab_size,
                            num_layers=args.num_layers,
                            dropout_ratio=args.dropout_ratio)

    decoder_model = nn.DataParallel(decoder_model)
    decoder_model.load_state_dict(torch.load(args.decoder_model_path))

    # move to GPU and evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: {}".format(device))
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)

    return encoder_model, decoder_model


def prediction(args):
    # load test data
    test_df = pd.read_csv(args.test_data_path)
    for i, path in enumerate(test_df["file_name"]):
        test_df.iloc[i]["file_name"] = os.path.join(args.root_img_dirc, path)

    # load vocabulary
    vocab = pickle.load(open(args.vocab_path, "rb"))

    # set prediction model
    encoder_model, decoder_model = model_setting(args)
    encoder_model.eval()
    decoder_model.eval()

    # split character level
    opinion_lst = list(test_df["caption"])
        
    # initialize
    img_path_lst = list(test_df["file_name"])
    opinions_lst = []
    captions_lst = []
    transform = set_transform(resize=(224, 224), crop_size=None, horizontal_flip=False, normalize=True)
    rm_path_cnt = 0

    # prediction
    with torch.no_grad():
        for img_path, op in tqdm(zip(img_path_lst, opinion_lst), total=len(img_path_lst)):
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = transform(img)
                img = to_variable(img)
                fea = encoder_model(img.unsqueeze(0))
                fea = fea.view(fea.size(0), args.vis_dim, args.vis_num).transpose(1,2)

                ids, _ = decoder_model.module.beam_search_captioning(fea, vocab, beam_size=args.beam_size)
                
                opinions_lst.append(op.lower())
                captions_lst.append(ids)
            else:
                rm_path_cnt += 1

    # decode captions
    decoded_cap_lst = decode_caption(captions_lst, vocab.idx2word)

    # save prediction result
    save_prediciton(opinions_lst, decoded_cap_lst, args.save_dirc)


def make_parse():
    parser = argparse.ArgumentParser(
        prog="prediction.py",
        usage="predict by trained model",
        description="description",
        epilog="end",
        add_help=True
    )

    # data Argument
    parser.add_argument("--root_img_dirc", type=str, default="../images/val2014")
    parser.add_argument("--test_data_path", type=str, default="../data/label/val.csv")
    parser.add_argument("--vocab_path", type=str, default="../data/vocab/vocab.pkl")
    parser.add_argument("--encoder_model_path", type=str, default="../data/model/encoder.pth")
    parser.add_argument("--decoder_model_path", type=str, default="../data/model/decoder.pth")
    parser.add_argument("--save_dirc", type=str, default="../data/prediction")

    # params Argument
    parser.add_argument("--vis_dim", type=int, default=2048)
    parser.add_argument("--vis_num", type=int, default=196)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout_ratio", type=float, default=0.0)
    parser.add_argument("--beam_size", type=int, default=1)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = make_parse()
    prediction(args)
