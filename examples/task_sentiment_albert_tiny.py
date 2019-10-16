#! -*- coding:utf-8 -*-
# 情感分析类似，加载albert_zh权重(https://github.com/brightmart/albert_zh)

import json
import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping

from bert4keras.bert import load_pretrained_model, set_gelu
from bert4keras.train import PiecewiseLinearLearningRate
from bert4keras.utils import SimpleTokenizer, load_vocab

set_gelu('tanh')  # 切换gelu版本

max_len = 100
config_path = '../albert_tiny/albert_config_tiny.json'
checkpoint_path = '../albert_tiny/albert_model.ckpt'
dict_path = '../albert_tiny/vocab.txt'
CONFIG = {
    'max_len': 128,
    'batch_size': 8,
    'epochs': 32,
    'use_multiprocessing': True,
    'model_dir': os.path.join('../model_files/bert'),
}

train_message = pd.read_csv('../data/Train_Data.csv', header=None).values.tolist()
chars = {}

data = []
# id,title,text,entity,negative,key_entity


for feather_data in train_message:
    data.append((feather_data[2], feather_data[4]))
    if feather_data[2]:
        for c in feather_data[2]:
            chars[c] = chars.get(c, 0) + 1

chars = {i: j for i, j in chars.items() if j >= 4}

_token_dict = load_vocab(dict_path)  # 读取词典
token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
    token_dict[c] = len(token_dict)
    keep_words.append(_token_dict[c])

for c in chars:
    if c in _token_dict:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])

tokenizer = SimpleTokenizer(token_dict)  # 建立分词器

if not os.path.exists('./random_order.json'):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('./random_order.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('./random_order.json'))

# 按照9:1的比例划分训练集和验证集
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:max_len]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

model = load_pretrained_model(
    config_path,
    checkpoint_path,
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
    albert=True
)

output = Lambda(lambda x: x[:, 0])(model.output)
output = Dense(1, activation='sigmoid')(output)
model = Model(model.input, output)
save = ModelCheckpoint(
    os.path.join(CONFIG['model_dir'], 'bert.h5'),
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='auto'
)
early_stopping = EarlyStopping(
    monitor='val_acc',
    min_delta=0,
    patience=8,
    verbose=1,
    mode='auto'
)
callbacks = [save, early_stopping]

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
    metrics=['accuracy']
)
model.summary()

train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=10,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    callbacks=callbacks
)


def predict(model, test_data):
    """
    预测
    :param test_data:
    :return:
    """
    X1 = []
    X2 = []
    for s in test_data:
        x1, x2 = tokenizer.encode(first=s[:CONFIG['max_len']])
        X1.append(x1)
        X2.append(x2)
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    predict_results = model.predict([X1, X2])
    return predict_results


test_data = pd.read_csv(os.path.join('data/Test_Data.csv'), encoding='utf-8')
predict_test = []
for i in test_data['text']:
    if i is not None:
        predict_test.append(str(i))
predict_results = predict(model, predict_test)
with open(os.path.join('data/bert/food-predict.csv'), 'w') as f:
    f.write("id,negative,key_entity\n")
    for i in range(test_data.shape[0]):
        label = 1 if predict_results[i][0] > 0.5 else 0
        if label == 1:
            f.write(test_data.id[i] + ',' + str(label) + ',' + test_data.entity[i] + '\n')
        else:
            f.write(test_data.id[i] + ',' + str(label) + '\n')
