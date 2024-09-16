import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset

import spacy
import jieba
import random
import math
import time
import pandas as pd
import numpy as np

spacy_en = spacy.load('en_core_web_sm')


def tokenize_zh(sentence):
    """
    中文分词
    """
    return [tok for tok in jieba.lcut(sentence)]


def tokenize_en(sentence):
    """
    英文分词
    """
    return [tok.text for tok in spacy_en.tokenizer(sentence)]


SRC = Field(tokenize=tokenize_zh,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)


# 定义字段映射，与 JSON 数据中的键对应
fields = {'chinese': ('src', SRC), 'english': ('trg', TRG)}

# 使用 TabularDataset 加载 JSON 格式的数据
train_data, valid_data, test_data = TabularDataset.splits(
    path='/Users/simo/PycharmProjects/translationLearn/translate/data',  # 数据文件所在的目录
    train='translation2019zh_train.json',
    validation='translation2019zh_valid.json',
    test='translation2019zh_test.json',
    format='json',
    fields=fields)


# 与之前相同
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)


BATCH_SIZE = 128

device = torch.device('mps')

# 使用 BucketIterator 创建迭代器
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device)


# 编码器（Encoder）
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded: [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [src_len, batch_size, hid_dim]
        return hidden, cell


# 解码器（Decoder）
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size]
        input = input.unsqueeze(0)
        # input: [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        # embedded: [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [1, batch_size, hid_dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction: [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # 初始化输出张量
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 编码器的最后隐藏状态作为解码器的初始隐藏状态
        hidden, cell = self.encoder(src)

        # 第一个输入是 <sos> token
        input = trg[0, :]

        for t in range(1, trg_len):
            # 将单词输入解码器
            output, hidden, cell = self.decoder(input, hidden, cell)
            # 保存预测结果
            outputs[t] = output
            # 决定是否使用教师强制
            teacher_force = random.random() < teacher_forcing_ratio
            # 选取最高概率的词作为预测
            top1 = output.argmax(1)
            # 下一步的输入
            input = trg[t] if teacher_force else top1

        return outputs


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)
        # output: [trg_len, batch_size, output_dim]
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        # 忽略第一个 <sos> token

        loss = criterion(output, trg)

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # 评估时不使用教师强制

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


model.apply(init_weights)
optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'cn_en_model.pt')

    print(f'第 {epoch + 1} 轮训练:')
    print(f'\t训练损失: {train_loss:.3f}')
    print(f'\t验证损失: {valid_loss:.3f}')


model.load_state_dict(torch.load('cn_en_model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'测试损失: {test_loss:.3f}')