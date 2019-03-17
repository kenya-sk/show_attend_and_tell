import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict


def up_sampling(train_path, root_dirc, thresh, word_freq):
    # load trainning data
    train_df = pd.read_csv(train_path)

    # count frequency of each caption
    train_cap2cnt = defaultdict(int)
    for cap in train_df["caption"]:
        train_cap2cnt[cap] += 1

    # NEED FIX 冗長なので関数に切り分ける
    # upsampling 
    # 異常なし：異常あり＝1:1
    up_train_dctlst = {"path":[], "caption":[]}
    train_cap2cnt.pop("異常なし")
    # 異常ありの各所見をupsampling
    for key in tqdm(train_cap2cnt.keys()):
        tmp_df = train_df[train_df["caption"] == key].reset_index()
        if len(tmp_df) < word_freq:
            copy_num = int(word_freq/len(tmp_df))
            for _ in range(copy_num):
                for i in range(len(tmp_df)):
                    up_train_dctlst["path"].append(tmp_df["path"][i])
                    up_train_dctlst["caption"].append(tmp_df["caption"][i])
            for i in range(int(word_freq - copy_num*len(tmp_df))):
                up_train_dctlst["path"].append(tmp_df["path"][i])
                up_train_dctlst["caption"].append(tmp_df["caption"][i])
        else:
            for i in range(len(tmp_df)):
                if (os.path.exists(os.path.join(root_dirc, tmp_df["path"][i]))):
                    up_train_dctlst["path"].append(tmp_df["path"][i])
                    up_train_dctlst["caption"].append(tmp_df["caption"][i])

    anormal_cnt = len(up_train_dctlst["path"])
    normal_df = train_df[train_df["caption"] == "異常なし"].reset_index()
    img_exist_cnt = 0

    # 異常なし >= 異常あり
    if len(normal_df) > anormal_cnt:
        for i in range(len(normal_df)):
            if (os.path.exists(os.path.join(root_dirc, normal_df["path"][i]))):
                up_train_dctlst["path"].append(normal_df["path"][i])
                up_train_dctlst["caption"].append(normal_df["caption"][i])
                img_exist_cnt += 1

            # whether the dataset is well-balanced
            if img_exist_cnt == anormal_cnt:
                break

    # 異常なし < 異常あり
    else:
        copy_num = int(anormal_cnt/len(normal_df))
        for _ in range(copy_num):
            for i in range(len(normal_df)):
                up_train_dctlst["path"].append(normal_df["path"][i])
                up_train_dctlst["caption"].append(normal_df["caption"][i])
        for i in range(int(anormal_cnt - copy_num*len(normal_df))):
            up_train_dctlst["path"].append(normal_df["path"][i])
            up_train_dctlst["caption"].append(normal_df["caption"][i])

    up_train_df = pd.DataFrame(up_train_dctlst)

    # balance = 1 mean, 異常なし:異常あり = 1:1
    balance = len(up_train_df[up_train_df["caption"] == "異常なし"])/len(up_train_df[up_train_df["caption"] != "異常なし"])
    print("異常なし/異常あり = {0}".format(balance))

    # shuffle dataset
    up_train_df = up_train_df.sample(frac=1).reset_index(drop=True)

    # save upsampling dataset
    up_train_df.to_csv("../data/label/upsampling/image_freq_thresh_{0}_each_word_{1}_train.csv".format(thresh, word_freq), index=False)
    print("DONE: save upsampling data")


if __name__ == "__main__":
    thresh = 5
    train_path = "../data/label/image_freq_thresh_{0}_train.csv".format(thresh)
    root_dirc = "/root/userspace/public/JSRT/sakka/medical_image_attention/image/jpg/"
    word_freq = 100
    up_sampling(train_path, root_dirc, thresh, word_freq)

