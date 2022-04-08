# code configs
import os
import torch
from transformers import BertConfig

# 超参数
# 词向量维度,与AlBERT词向量一致
# TODO:AlBERT的词向量和隐层数目不同


input_size = 768
# 隐藏层数目
hidden_size = 768
# lstm层数
lstm_layer = 2
# 训练周期
epoch = 30
# batch大小
batch_size = 64
# 学习率
learning_rate = 3e-5
# 权重衰减
weight_decay = 0.01
# 梯度裁剪
clip_grad = 5
# loss差值的patience
patience = 0.0002
# 耐心等待的epoch数
patience_num = 10

# 路径
data_dir = os.getcwd() + '/data/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
pretrained_model = 'pretrained_models/AnchiBERT'
model_save_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_save_dir + 'train.log'
bad_case_dir = os.getcwd() + '/bad_case/bad_case.txt'

# 分割比例
dev_split_size = 0.2

# 设备
gpu = ''
device = torch.device(f"cuda:{gpu}") if gpu != '' else torch.device("cpu")

# 标签
labels = ['image']
label2id = {
    "O": 0,
    "B-image": 1,
    "I-image": 2,
    "E-image": 3,
    "S-image": 4
}
id2label = {index: label for label, index in list(label2id.items())}
