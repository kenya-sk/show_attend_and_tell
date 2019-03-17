import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict


def split_dataset(data_df, thresh, save_dataset_dirc):
    """
    Divide the dataset while maintaining the original distribution.
    """
    
    # count frequency each caption
    cap2cnt = defaultdict(int)
    for cap in data_df["caption"]:
        cap2cnt[cap] += 1

    # split train, val, and test dataset
    # the ratio is (train:val:test = 8:1:1)
    train_dctlst = {"caption":[], "path":[]}
    val_dctlst = {"caption":[], "path":[]}
    test_dctlst = {"caption":[], "path":[]}
    for key, value in tqdm(cap2cnt.items()):
        val_num = int(value*0.1) if int(value*0.1) >= 1 else 1
        path_lst = list(data_df[data_df["caption"] == key]["path"])
        
        # add caption
        val_dctlst["caption"].extend([key for _ in range(val_num)])
        test_dctlst["caption"].extend([key for _ in range(val_num)])
        train_dctlst["caption"].extend([key for _ in range(len(path_lst) - 2*val_num)])
        
        # add path
        val_dctlst["path"].extend(path_lst[:val_num])
        test_dctlst["path"].extend(path_lst[val_num:2*val_num])
        train_dctlst["path"].extend(path_lst[2*val_num:])
    
    print("The Number of Train Data : {0}".format(len(train_dctlst["caption"])))
    print("The Number of Val Data   : {0}".format(len(val_dctlst["caption"])))
    print("The Number of Test Data  : {0}".format(len(test_dctlst["caption"])))
 
    pd.DataFrame(train_dctlst).to_csv("{0}/image_freq_thresh_{1}_train.csv".format(save_dataset_dirc, thresh), index=False)
    pd.DataFrame(val_dctlst).to_csv("{0}/image_freq_thresh_{1}_val.csv".format(save_dataset_dirc, thresh), index=False)
    pd.DataFrame(test_dctlst).to_csv("{0}/image_freq_thresh_{1}_test.csv".format(save_dataset_dirc, thresh), index=False)
    print("DONE: save datasets ({0})".format(save_dataset_dirc))


if __name__ == "__main__":
    thresh = 5
    root_img_dirc = "/root/userspace/public/JSRT/sakka/medical_image_attention/image/jpg"
    data_df = pd.read_csv("../data/label/image_freq_thresh_{0}.csv".format(thresh))
    data_df = check_img_path(root_img_dirc, data_df)

    save_dataset_dirc = "../data/label"
    split_dataset(data_df, thresh, save_dataset_dirc)

