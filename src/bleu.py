import sys
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk import bleu_score
import logging
import argparse

from utils import Vocabulary


def bleu(pred, answer, mode="1-gram"):
    if mode == "1-gram":
        weights = [1.0]
    elif mode == "2-gram":
        weights = [0.5, 0.5]
    elif mode == "3-gram":
        weights = [0.3, 0.3, 0.3]
    elif mode == "4-gram":
        weights = [0.25, 0.25, 0.25, 0.25]
    else:
        sys.stdout.write("Not support mode")
        sys.exit()
        
    return bleu_score.sentence_bleu([pred], answer, weights=weights)


def evaluate(args, logger):
    pred_df = pd.read_csv(args.pred_path)
    for mode_idx in range(4):
        # calculate 1-4 gram
        mode = "{0}-gram".format(mode_idx+1)
        score_lst = []
        for i in range(len(pred_df)):
            pred_lst = []
            pred = pred_df["pred"][i]
            for p in pred:
                pred_lst.append(p)
            
            ans_lst = []
            answers = word_tokenize(pred_df["ans"][i])[0]
            for ans in answers:
                ans_lst.append(ans)

            bleu_score = bleu(pred_lst, ans_lst, mode)
            score_lst.append(bleu_score)

        logger.debug("score of {0}: {1}".format(mode, np.mean(score_lst)))


def score_parse():
    parser = argparse.ArgumentParser(
        prog="bleu_score.py",
        usage="calculate bleu score",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--pred_path", type=str,
                       default="../data/prediction/prediction.csv")
    parser.add_argument("--vocab_path", type=str,
                        default="../data/vocab/vocab.pkl")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logger = logging.getLogger("__name__")
    log_path = "../logs/bleu.log"
    logging.basicConfig(filename=log_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    args = score_parse()
    logger.debug("Running with args: {0}".format(args))
    evaluate(args, logger)

