# show_attend_and_tell

This repository reimplement "Show, Attend and Tell" model.  
paper: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention  (https://arxiv.org/pdf/1502.03044.pdf)

## Getting Started

### Install Required Packages
First ensure that you have installed the following required packages.  
- Python 3.5.2
- NumPy 1.13.1
- Pandas 0.23.0
- Matplotlib 2.0.2
- OpenCV 3.3.0
- Pillow 5.1.0
- PyTorch 0.4.1
- Natural Language Toolkit (NLTK)  


### MS-COCO Data
 MS-COCO dataset downloaded in following path. Image format is ".jpg".
 ```
 Image Path: 
 /root/userspace/public/JSRT/sakka/ms_coco_sakka_attention/data/images/train_2014 or val_2014/

 Caption (Answer label):
 /root/userspace/public/JSRT/sakka/ms_coco_sakka_attention/data/annotations/captions_train2014.json
 ```

If downloading by yourself, both training and validation are necessary.  
Downloding from here.  
http://cocodataset.org/#download