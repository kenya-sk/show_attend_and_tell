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
from utils import  to_variable, Vocabulary, decode_caption


def set_pred_transform():
    # NEED FIX 
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    return transform


def save_prediciton(names_lst, captions_lst, save_dirc, thresh, each_word):
    # get pred and test dataframe
    pred_dct_lst = {"pred": [], "ans": []}
    for i in range(len(names_lst)):
        pred_dct_lst["ans"].append(names_lst[i].replace(" ", ""))
        pred_dct_lst["pred"].append(captions_lst[i])

    pd.DataFrame(pred_dct_lst).to_csv(
        "{0}/image_freq_thresh_{1}_each_word_{2}_beam1.csv".format(save_dirc, thresh, each_word), index=False)


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


def conv_character_level(caption_lst):
    name_lst = []
    for cap in caption_lst:
        name = ""
        for char in cap:
            name += "{} ".format(char)
        name_lst.append(name)
    return name_lst


def prediction(args):
    # load test data
    test_df = pd.read_csv(args.test_data_path)
    for i, path in enumerate(test_df["path"]):
        test_df.iloc[i]["path"] = args.root_img_dirc + path

    # load vocabulary
    vocab = pickle.load(open(args.vocab_path, "rb"))

    # set prediction model
    encoder_model, decoder_model = model_setting(args)
    encoder_model.eval()
    decoder_model.eval()

    # split character level
    opinion_lst = list(test_df["caption"])
    name_lst = conv_character_level(opinion_lst)
        
    # initialize
    img_path_lst = list(test_df["path"])
    names_lst = []
    captions_lst = []
    alphas_lst = []
    transform = set_pred_transform()
    rm_path_cnt = 0

    # prediction
    with torch.no_grad():
        for img_path, name in tqdm(zip(img_path_lst, name_lst), total=len(img_path_lst)):
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = transform(img)
                img = to_variable(img)
                fea = encoder_model(img.unsqueeze(0))
                fea = fea.view(fea.size(0), args.vis_dim, args.vis_num).transpose(1,2)

                #ids, weights = decoder_model.module.captioning(fea)
                ids, weights = decoder_model.module.beam_search_captioning(fea, vocab, beam_size=1)
                
                names_lst.append(name)
                captions_lst.append(ids)
                alphas_lst.append(weights)
            else:
                rm_path_cnt += 1

    # decode captions
    decoded_cap_lst = decode_caption(captions_lst, vocab.idx2word)

    # save prediction result
    save_prediciton(names_lst, decoded_cap_lst, args.save_dirc, args.cap_freq_thresh, args.each_word)


def make_parse():
    parser = argparse.ArgumentParser(
        prog="prediction.py",
        usage="predict by trained model",
        description="description",
        epilog="end",
        add_help=True
    )

    # data Argument
    parser.add_argument("--root_img_dirc", type=str,
                        default="/root/userspace/public/JSRT/sakka/medical_image_attention/image/jpg/")
    parser.add_argument("--test_data_path", type=str,
                        default="/root/userspace/public/JSRT/sakka/medical_image_attention/data/label/image_freq_thresh_5_test.csv")
    parser.add_argument("--vocab_path", type=str,
                        default="/root/userspace/public/JSRT/sakka/medical_image_attention/data/vocab/vocab_freq_thresh_5.pkl")
    parser.add_argument("--encoder_model_path", type=str,
                        default="/root/userspace/public/JSRT/sakka/medical_image_attention/data/model/encoder_freq5_test.pth")
    parser.add_argument("--decoder_model_path", type=str,
                        default="/root/userspace/public/JSRT/sakka/medical_image_attention/data/model/decoder_freq5_test.pth")
    parser.add_argument("--save_dirc", type=str,
                        default="../data/prediction")

    # params Argument
    parser.add_argument("--vis_dim", type=int, default=2048)
    parser.add_argument("--vis_num", type=int, default=196)
    parser.add_argument("--embed_dim", type=int, default=118)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=118)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout_ratio", type=float, default=0.0)
    parser.add_argument("--cap_freq_thresh", type=int, default=5)
    parser.add_argument("--each_word", type=int, default=100)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = make_parse()
    prediction(args)
