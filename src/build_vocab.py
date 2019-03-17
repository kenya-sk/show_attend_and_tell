import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

from utils import Vocabulary


def build_vocab(word_lst):
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    for word in word_lst:
        vocab.add_word(word)
        
    return vocab


def max_len(cap_lst):
    max_len = 0
    max_cap = ""
    for cap in cap_lst:
        if max_len < len(cap):
            max_len = len(cap)
            max_cap = cap
            
    print("Max Caption: {0}".format(max_cap))
    print("Max Length: {0}".format(max_len))


def build(cap_freq_thresh, char_min_freq, save_vocab_dirc):
    data_df = pd.read_csv("../data/label/image_freq_thresh_{0}.csv".format(cap_freq_thresh))
    cap_lst = list(data_df["caption"])
    max_len(cap_lst)

    # freq of each character
    char2cnt = defaultdict(int)
    for cap in cap_lst:
        for char in cap:
            char2cnt[char] += 1

    # extract top frequancy character
    char_min_freq = 0
    top_char_lst = []
    for char, cnt in sorted(char2cnt.items(), key=lambda x: -x[1]):
        if cnt < char_min_freq:
            break
        top_char_lst.append(char)

    # bulid vocabulary
    vocab = build_vocab(top_char_lst)
    vocab_path = "{0}/vocab_freq_thresh_{1}.pkl".format(save_vocab_dirc, cap_freq_thresh)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print("Number of  vocablary: {0}".format(len(vocab)))
    print("Save the vocabulary wrapper to {0}".format((vocab_path)))


if __name__ == "__main__":
    cap_freq_thresh = 5
    char_min_freq = 0
    save_vocab_dirc = "../data/vocab"
    build(cap_freq_thresh, char_min_freq, save_vocab_dirc)

