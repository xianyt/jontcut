# jontcut
JointCut: A Simple and Efficient Neural Model for Joint Thai Word and Syllable Segmentation

# Requirements and Installation
A computer running macOS or Linux. You can install requirements by following commands:

`
$ pip install -r requirements.txt 
`

# Training
Use python train.py to train a new word segmentation model, with two steps:

##1. preprocess

> $ python preprocess.py --data_path data/BEST_2010 --output_path data/best_2010_csv

> $ python preprocess.py --data_path data/BEST_2010/TEST.tx --output_path data/best_2010_csv/test.csv --split_ratio 0.0


##2. training model

Training JointCut-Base model with CUDA support:

> $ CUDA_VISIBLE_DEVICES=0 python train.py \
   --train data/best_2010_csv/train \
   --valid data/best_2010_csv/valid \
   --cache cache/best_2010_all \
   --output output/base.pt \
   --char_embed_dim 32 \
   --type_embed_dim 16 \
   --transformer_d_model 48 \
   --transformer_n_head 6 \
   --transformer_num_layers 1  \
   --transformer_dim_feedforward 128  \
   --syllable_dense_dim 25 \
   --word_dense_dim 100 \
   --syllable_loss_lambda 0.5
 
# Use pre-trained model

> python joint_cut.py --model models/base.pt --in_file test_file/t1.txt --out_file test_file/t1_tok.txt
