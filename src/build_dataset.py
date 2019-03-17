import numpy as np
import pandas as pd
import argparse

from utils import check_img_path
from clean_dataset import clean_dataset
from split_dataset import split_dataset
from up_sampling import up_sampling
from build_vocab import build

def make_dataset(args):
    # preprrocessing caption
    clean_dataset(args.label_path, args.save_label_dirc, args.cap_freq_thresh)
    
    # split into train, val, and test while keeping the original distribution
    data_df = pd.read_csv("{0}/image_freq_thresh_{1}.csv".format(args.save_label_dirc, args.cap_freq_thresh))
    data_df = check_img_path(args.root_img_dirc, data_df)
    split_dataset(data_df, args.cap_freq_thresh, args.save_label_dirc)

    # upsampling training data
    train_data_path = "{0}/image_freq_thresh_{1}_train.csv".format(args.save_label_dirc, args.cap_freq_thresh)
    up_sampling(train_data_path, args.root_img_dirc, args.cap_freq_thresh, args.each_word)

    # build vocabulary
    build(args.cap_freq_thresh, args.char_min_freq, args.save_vocab_dirc)



def make_parse():
    parser = argparse.ArgumentParser(
        prog="prediction.py",
        usage="predict by trained model",
        description="description",
        epilog="end",
        add_help=True
    )

    # data Argument
    parser.add_argument("--label_path", type=str,
                        default="../data/label/all_label.csv")
    parser.add_argument("--save_label_dirc", type=str,
                        default="../data/label")
    parser.add_argument("--root_img_dirc", type=str,
                        default="/root/userspace/public/JSRT/sakka/medical_image_attention/image/jpg")
    parser.add_argument("--save_vocab_dirc", type=str,
                        default="../data/vocab")

    # params Argument
    parser.add_argument("--cap_freq_thresh", type=int, default=5)
    parser.add_argument("--each_word", type=int, default=100)
    parser.add_argument("--char_min_freq", type=int, default=0)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = make_parse()
    make_dataset(args)
