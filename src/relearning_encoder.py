import numpy as np
import pandas as pd
import datetime
import argparse
import logging
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

from utils import tensor2numpy, to_variable, set_transform
from model import Resnet
from data_loader import get_relearning_loader


logger = logging.getLogger(__name__)
log_path = "/root/userspace/public/JSRT/sakka/medical_image_attention/logs/relearning_encoder.log"
logging.basicConfig(filename=log_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def get_dataset(args):
    label_df = pd.read_csv(args.relearning_label_path)
    X = np.array(label_df["ファイル名"])
    y = np.array(label_df["所見"])

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)
    logger.debug("Train: {}".format(len(train_X)))
    logger.debug("Val: {}".format(len(val_X)))

    return train_X, val_X, train_y, val_y


def relearning_encoder(args):
    train_X, val_X, train_y, val_y = get_dataset(args)

    # make dataset
    transform = set_transform(args.resize, args.crop_size)

    # DataLoader化
    train_loader = get_relearning_loader(args.root_img_dirc, args.train_data_path, args.class_num, transform, args.batch_size, args.shuffle)
    val_loader = get_relearning_loader(args.root_img_dirc, args.val_data_path, args.class_num, transform, args.batch_size, args.shuffle)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("DEVICE: {}".format(device))

    train_loss_lst = []
    val_loss_lst  = [10000]
    not_improved_count = 0

    # define model
    model = Resnet(args.class_num).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                    model.parameters()), lr=args.lr)

    # learning
    logger.debug("START: Train")
    st = datetime.datetime.now()
    for epoch in range(args.num_epochs):
        model.train()
        train_total_loss = 0
        val_total_loss = 0
        for train_data in tqdm(train_loader, desc="Train Data"):
            train_x, train_label = Variable(train_data["img"].to(device)), Variable(train_data["label"].to(device))
            optimizer.zero_grad()
            y_ = model(train_x.float()).float()
            loss = criterion(y_, train_label.long())
            train_total_loss += float(loss)
            loss.backward()
            optimizer.step()
            
        train_loss_lst.append(train_total_loss)
        del train_x, train_label, y_, loss
        gc.collect()
            
        # validation and early stopping
        model.eval()
        for val_data in tqdm(val_loader, desc="Val Data"):
            val_x, val_label = Variable(val_data["img"].to(device)), Variable(val_data["label"].to(device))
            val_y_ = model(val_x.float()).float()
            val_loss = criterion(val_y_, val_label)
            val_total_loss += float(val_loss)
            
        val_loss_lst.append(val_total_loss)
        if val_loss_lst[-1] >= val_loss_lst[-2]:
            not_improved_count += 1
        else:
            not_improved_count = 0
        
        if not_improved_count == 0:
            # save best model
            torch.save(model.state_dict(), args.save_model_path)
            
        if (not_improved_count >= args.stop_count) and (epoch > args.min_epoch):
            logger.debug("Early Stopping")
            ed = datetime.datetime.now()
            logger.debug("epoch: {0}, train loss: {1}, val loss: {2}, time: {3}".format(epoch+1, train_total_loss, val_total_loss, ed-st))
            train_loss_lst.append(train_total_loss)
            break
            
        del val_x, val_label, val_loss
        gc.collect()
            
        # log
        ed = datetime.datetime.now()
        logger.debug("epoch: {0}, train loss: {1}, val loss: {2}, time: {3}".format(epoch+1, train_total_loss, val_total_loss, ed-st))
        st = datetime.datetime.now()


def make_parse():
    parser = argparse.ArgumentParser(
        prog="relearning_encoder.py",
        usage="relearning of encoder model by multi-class classification",
        description="description",
        epilog="end",
        add_help=True
    )

    # data Argument
    parser.add_argument("--relearning_label_path", type=str, 
        default="/root/userspace/public/JSRT/sakka/medical_image_attention/data/label/relearning_cnn_num_label.csv")
    parser.add_argument("--root_img_dirc", type=str, 
        default="/root/userspace/public/JSRT/sakka/medical_image_attention/image/jpg")
    parser.add_argument("--train_data_path", type=str, 
        default="/root/userspace/public/JSRT/sakka/medical_image_attention/data/label/relearning_cnn_train.csv")
    parser.add_argument("--val_data_path", type=str, 
        default="/root/userspace/public/JSRT/sakka/medical_image_attention/data/label/relearning_cnn_val.csv")
    parser.add_argument("--save_model_path", type=str,
        default="/root/userspace/public/JSRT/sakka/medical_image_attention/data/model/model.pth")


    # params Argument
    parser.add_argument("--resize", type=tuple, default=(256, 256))
    parser.add_argument("--crop_size", type=tuple, default=(224, 224))
    parser.add_argument("--class_num", type=int, default=11)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--min_epoch", type=int, default=5)
    parser.add_argument("--stop_count", type=int, default=2)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = make_parse()
    logger.debug("Running with args: {0}".format(args))
    relearning_encoder(args)
