# show_attend_and_tell

This repository reimplement "Show, Attend and Tell" model.  
paper: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention  (https://arxiv.org/pdf/1502.03044.pdf)

Adittions:
- Clip Gradient
- Beam Search

### Getting Started

## Install Required Packages
First ensure that you have installed the following required packages.  
```
pip install -r requirements.txt
```

- python 3.5.2
- numPy 1.13.1
- pandas 0.23.0
- matplotlib 2.0.2
- pillow 5.1.0
- PyTorch 0.4.1
- tqdm 4.24.0
- Natural Language Toolkit (NLTK) 3.2.4

## Weight of Trained Model
If you need weights of trained model, please download below shared URL. The Data size is about 320.5 MB.


## MS-COCO Data
MS-COCO dataset downloaded in following path. Image format is ".jpg".
```
Image Path: 
show_attend_and_tell/data/images/train_2014 or val_2014/

Caption (Answer label):
show_attend_and_tell/data/annotations/captions_train2014.json
```

If downloading by yourself, both training and validation are necessary.  
Downloding from here (http://cocodataset.org/#download).
```
Train data:
wget http://images.cocodataset.org/zips/train2014.zip

Validataion data:
wget http://images.cocodataset.org/zips/val2014.zip

Caption data:
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

## Prepare the Dataset
Build datasets with columns of image paths and corresponding captions.   
Each datasets are saved in csv format under the directory of "../data/label/"
```
# build train and validation datasets

python3 ./src/build_dataset.py
```

## Prepare the Vocaburary
Based on the frequency of words appearing in training data, build vocaburary to use model. default vacaburary size is 10,000.  
The vocaburary is saved in "../data/vocab/vocab.pkl".
```
# build vocaburary

python3 ./src/build_vocaburay.py
```

## Training Encoder-Decoder Model
In training, encoder and decoder are learned at the save time based on cross entropy loss of caption and loss of attention mechanism.
```
1. ./src/train.py [-h]  [--root_img_dirc]  [--train_data_path]  
                  [--vak_data_path]  [--vaocb_path]  [--save_encoder_path]
                  [--save_decoder_path]  [--fine_tune_encoder]  [--resize]  [--crop_size]  [--shuffle]  [--num_workers]
                  [--vis_dim]  [--vis_num]  [--embed_dim]  [--hidden_dim]
                  [--alpha_c]  [--vocab_size]  [--num_layers]  [--dropout_rate]
                  [--encoder_lr]  [--decoder_lr]  [--num_epochs]  [--min_epoch]]
                  [--batch_size]  [--stop_count]  [--grad_clip]  [--beam_size]
```

Hyper parameters and learning conditions can be set using arguments. Details are as follows.
```
[--root_img_dirc]            : The directory of raw image
[--train_data_path]          : The path of training dataset
[--val_data_path]            : The path of validataion dataset
[--vocab_path]               : The path of vocabuary 
[--save_encoder_path]        : The path of save the weight of trained encoder
[--save_decoder_path]        : The path of save the weight of treined decoder
[--fine_tune_encoder]        : Fine tuning encoder model (True or False)
[--resize]                   : Size for resizing the image before random crop (tuple)
[--crop_size]                : Image size to be input to encoder (tuple)
[--shuffle]                  : Whether to shuffle the dataset at training
[--num_workers]              : The number of threads used during training
[--vis_dim]                  : The number of input dimensions of decoder
[--vis_num]                  : The numebr of input channel of decoder
[--embed_dim]                : The dimension of emnedding
[--hidden_dim]               : The number of dimension of hidden layer of LSTM
[--alpha_c]                  : The ratio of attention mechanism on loss
[--vocab_size]               : The number of vocabulay to use
[--num_layers]               : The number of LSTM layer
[--dropout_rate]             : Dropout rate of decoder
[--encoder_lr]               : The learning rate of encoder
[--decoder_lr]               : The learing rate of decoder
[--num_epochs]               : The maximum number of epoch
[--min_epoch]                : The minimum number of epoch
[--batch_size]               : Bach size
[--stop_count]               : If not_improved_count < stop_count, execute early stopping
[--grad_clip]                : To set the threshold of the gradient
[--beam_size]                : Search width of beam search
```

## Evaluation
Evaluate the test data with BLEU score using the trained model. The procedure is as follows.

```
# Set path of trained model and path of test data.
# output is 2 columns csv file. (col1, col2) = ("pred", "ans")
1. ./src/prediction.py [-h]  [--root_img_dirc]  [--test_data_path]  
                       [--vocab_path]  [--encoder_model_path]  [--decoder_model_path]  [--save_dirc]
                       [--vis_dim]  [--vis_num]  [--embed_dim]  
                       [--hidden_dim]  [--vocab_size]  [--num_layers]  [--dropout_ratio]  [--beam_size]

# If you want to see the result visually, please use jupyter notebook
2. ./src/visualize.ipynb

# Evaluate using the csv file calculated by the above procedure.
# The metrics is BLEU 1-4
3. ./src/bleu.py
```

Hyper parameters of prediction model can be set using arguments. Details are as follows.
```
[--root_img_dirc]           : The directory of raw image
[--test_data_path]          : The path of test dataset
[--vocab_path]              : The path of vocabuary 
[--encoder_model_path]      : The path of trained encoder model
[--decoder_model_path]      : The path of treined decoder
[--save_dirc]               : The path of save the prediction result
[--vis_dim]                 : The number of input dimensions of decoder (the same number as training)
[--vis_num]                 : The numebr of input channel of decoder (the same number as training)
[--embed_dim]               : The dimension of emnedding (the same number as training)
[--hidden_dim]              : The number of dimension of hidden layer of LSTM (the same number as training)
[--vocab_size]              : The number of vocabulay to use (the same number as training)
[--num_layers]              : The number of LSTM layer (the same number as training)
[--dropout_rate]            : Not use dropout (set 0.0)
[--beam_size]               : Search width of beam search
```

# Result
## BLUE score

## Visualize