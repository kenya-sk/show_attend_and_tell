import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict


def rm_brackets(text):
    start_idx = max(text.find("（"), text.find("("))
    end_idx = max(text.find("）"),  text.find(")"))

    if (start_idx >= 0) and (end_idx >= 0):
        replace_wd = text[start_idx:end_idx+1]
        text = text.replace(replace_wd, "")
   
    return text


def cap_cnt(cap_lst):
    cap2cnt = defaultdict(int)
    for cap in cap_lst:
        cap2cnt[cap] += 1
        
    return cap2cnt


def set_convert_dict():
    convert_dict = OrderedDict()
    convert_dict["前回と著変なし"] = ""
    convert_dict["変化なし"] = ""
    convert_dict["CT上問題なし"] = ""
    convert_dict["手入力　部位,"] = ""
    convert_dict["ペースメーカー植込み後"] = "デバイス植込後"
    convert_dict["ペースメーカー挿入後"] = "デバイス植込後"
    convert_dict["ペースメーカー埋込み後"] = "デバイス植込後"
    convert_dict["ペースメーカー植え込み後"] = "デバイス植込後"
    convert_dict["ペースメーカー挿入"] = "デバイス植込後"
    convert_dict["ICDあり"] = "デバイス植込後"
    convert_dict["異常所見なし"] = "異常なし"
    convert_dict["異常所見無し"] = "異常なし"
    convert_dict["正常、肺：異常所見なし"] = "異常なし"
    convert_dict["肺野に異常所見なし"] = "異常なし"
    convert_dict["肺野に異常なし"] = "異常なし"
    convert_dict["肺野に明らかな異常なし"] = "異常なし"
    convert_dict[" "] = ""
    convert_dict["　"] = ""
    convert_dict["の疑い"] = ""
    convert_dict["疑い"] = ""
    convert_dict["手術後"] = "術後"
    convert_dict["胸膜ゆ着"] = "胸膜癒着"
    convert_dict["非定型抗酸菌症所見"] = "非定型抗酸菌"
    
    return convert_dict


def clean_cap(cap):
    # remove bracket
    cap = rm_brackets(cap)
    
    # format the label
    convert_dict = set_convert_dict()
    for before, after in convert_dict.items():
        cap = cap.replace(before, after)
        if len(cap) == 0:
            return None
    
    return cap


def split_cap(cap, split_lst):
    for sp in split_lst:
        if sp in cap:
            return cap.split(sp)
    return [cap]


def extract_high_freq_cap(label_df, freq_thresh):
    # count each caption
    cap2cnt = cap_cnt(list(label_df["caption"]))

    # extract the caption whose frequency is equal to or larger than the threshold
    high_freq_label_dctlst = {"path": [], "caption": []}
    for i, cap in enumerate(label_df["caption"]):
        if (cap2cnt[cap] >= freq_thresh) and (cap is not None):
            high_freq_label_dctlst["path"].append(label_df["path"][i])
            high_freq_label_dctlst["caption"].append(label_df["caption"][i])
    high_freq_label_df = pd.DataFrame(high_freq_label_dctlst)
    
    return high_freq_label_df


def print_cap_info(label_df, init_data_num):
    print("Coverd Rate    : {0}".format(len(label_df) / init_data_num))
    print("Normal Rate    : {0}".format(len(label_df[label_df["caption"] == "異常なし"]) / len(label_df)))
    print("Anormal Rate   : {0}".format(len(label_df[label_df["caption"] != "異常なし"]) / len(label_df)))
    print("Unique Caption : {0}".format(len(np.unique(label_df["caption"]))))


def clean_dataset(label_path, save_dirc, freq_thresh=5):
    # load label data
    label_df = pd.read_csv(label_path)
    init_data_num = len(label_df)

    # clean each caption
    for i, cap in enumerate(label_df["caption"]):
        label_df["caption"][i] = clean_cap(cap)
    label_df = label_df.dropna().reset_index(drop=True)
        
    # create a dataset by splitting multiple captions
    clean_label_dct = {"path":[], "caption":[]}
    for i in range(len(label_df)):
        cap = label_df["caption"][i]
        cur_path = label_df["path"][i]
        split_lst = [",", "、"]
        split_cap_lst = split_cap(cap, split_lst)
        for s_cap in split_cap_lst:
            clean_label_dct["path"].append(cur_path)
            clean_label_dct["caption"].append(s_cap)
    label_df = pd.DataFrame(clean_label_dct)

    # extract the caption whose frequency is equal to or larger than the threshold
    high_freq_label_df = extract_high_freq_cap(label_df, freq_thresh)

    # output dataset infomation
    print_cap_info(high_freq_label_df, init_data_num)

    # save dataset of high frequency caption
    high_freq_label_df.to_csv("{0}/image_freq_thresh_{1}.csv".format(save_dirc, freq_thresh), index=False)
    print("DONE: clean dataset")


if __name__ == "__main__":
    label_path = "../data/label/feature_label.csv"
    save_dirc = "../data/label"
    clean_dataset(label_path, save_dirc, freq_thresh=5)
