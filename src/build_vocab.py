import os
import json
import pickle
from tqdm import trange
from collections import defaultdict
from tqdm import trange
import nltk
nltk.download('punkt')

from utils import Vocabulary


def build_vocab(word_lst, size=10000):
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    for word in word_lst[:size-4]:
        vocab.add_word(word)
        
    return vocab


def get_word_lst(annotations):
    word_lst = []
    word_cnt_dict = defaultdict(int)
    for i in trange(len(annotations["annotations"])):
        caption = annotations["annotations"][i]["caption"]
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        for token in tokens:
            word_cnt_dict[token] += 1
            if token not in word_lst:
                word_lst.append(token)

    return word_cnt_dict


def build(ann_file_path, save_vocab_dirc):
    # load annoteted data
    annotations = json.load(open(ann_file_path, "r"))

    # count the number of words
    word_cnt_dict = get_word_lst(annotations)

    # decreasing sort based on word frequency
    sorted_word_lst = []
    for k, _ in sorted(word_cnt_dict.items(), key=lambda x: -x[1]):
        sorted_word_lst.append(k)

    # bulid vocabulary
    vocab = build_vocab(sorted_word_lst)
    vocab_path = os.path.join(save_vocab_dirc, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print("Number of  vocablary: {0}".format(len(vocab)))
    print("Save the vocabulary wrapper to {0}".format((vocab_path)))


if __name__ == "__main__":
    ann_file_path = "../data/annotations/captions_train2014.json"
    save_vocab_dirc = "../data/vocab"
    build(ann_file_path, save_vocab_dirc)

