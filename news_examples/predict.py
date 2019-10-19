import os

import numpy as np
import pandas as pd
from bert4keras.utils import SimpleTokenizer, load_vocab
import keras

model = keras.models.load_model("../model_files/bert/bert.h5")

CONFIG = {
    'max_len': 256,
    'batch_size': 48,
    'epochs': 32,
    'use_multiprocessing': True,
    'model_dir': os.path.join('../model_files/bert'),
}
config_path = '../albert/bert_config.json'
checkpoint_path = '../albert/bert_model.ckpt'
dict_path = '../albert/vocab.txt'
_token_dict = load_vocab(dict_path)  # 读取词典
token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
    token_dict[c] = len(token_dict)
    keep_words.append(_token_dict[c])


tokenizer = SimpleTokenizer(_token_dict)  # 建立分词器


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


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


test_data = pd.read_csv(os.path.join('../data/Test_Data.csv'), encoding='utf-8')
predict_test = []
for i in test_data['text']:
    if i is not None:
        predict_test.append(str(i))
predict_results = predict(model, predict_test)
with open(os.path.join('../data/bert/news-predict.csv'), 'w') as f:
    f.write("id,negative,key_entity\n")
    for i in range(test_data.shape[0]):
        f.write(str(test_data.id[i]) + ',' + str(predict_results.argmax(axis=1)[0]) + '\n')
