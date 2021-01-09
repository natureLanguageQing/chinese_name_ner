#! -*- coding: utf-8 -*-

import json

from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

from bert4keras.backend import keras, K
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

basepath = '/Users/chenquanbao/Downloads/chinese_wwm_ext_L-12_H-768_A-12'
# basepath='/export/sdb/admin/train/chenquanbao/chinese_wwm_ext_L-12_H-768_A-12'
rbtl_config_path = os.path.join(basepath, 'bert_config.json')
rbtl_checkpoint_path = os.path.join(basepath, 'bert_model.ckpt')
rbtl_dict_path = os.path.join(basepath, 'vocab.txt')
wait_train_data = '../data/bmes_train.json'

max_text_length = 150
epochs = 1
batch_size = 16
bert_layers = 3
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率


def del_character(content: str):
    content = content.replace('[', '')
    return content


def get_d(contents: dict) -> (list, set):
    """
    根据文本和标注的信息，将文本的所有字段都进行标注
    :return:
    """
    word_list = []
    word_dict = {}
    for x in contents['entities']:
        word, flag = x.split('-')
        word = del_character(word)
        word_list.append(word)
        word_dict[word] = flag
    if len(word_list) == 0:
        return [[contents['text'], 'O']], set()
    # 去除特殊符号
    content = del_character(contents['text'])
    # 对提取的命名体根据长度 排序
    wordlista = [(word, flag, len(word)) for word, flag in word_dict.items()]
    wordlista = sorted(wordlista, key=lambda x: x[2], reverse=True)

    for i in range(0, len(wordlista)):
        content = content.replace(wordlista[i][0], '<<<' + str(i) + '>>>')

    rex = re.compile('<<<|>>>')
    res = rex.split(content)

    result = []
    for i in range(0, len(res)):
        if i % 2 == 0:
            if res[i] != '':
                result.append([res[i], 'O'])
        else:
            word = wordlista[int(res[i])][0]
            result.append([word, word_dict[word]])

    return result, set(word_dict.values())


def load_data_base(filename):
    D = []
    labels = set()
    f = open(filename, encoding='utf-8')
    medical = json.load(f)
    maxlen = 0
    for content in medical:
        maxlen = max(maxlen, len(content['text']))
        try:
            d, label = get_d(content)
            D.append(d)
            labels = labels.union(label)
        except Exception as e:
            print('maxerror-------------------------------------', content, e)

    return D, labels


def load_data(filename):
    D, labels = load_data_base(filename)
    labels = list(set(labels))
    return D, labels


def load_data_tri(filename):
    D, labels = load_data_base(filename)
    valid_data = []
    train_data = []
    test_data = []
    for index, data in enumerate(D):
        count = index % 6
        if count == 1:
            valid_data.append(data)
        elif count == 2:
            test_data.append(data)
        else:
            train_data.append(data)
    return train_data, valid_data, test_data, labels


def get_id2label(label_path, train_path):
    train_data, labels = load_data(train_path)
    id2label = dict(enumerate(labels))
    labels_file = open(label_path, "w", encoding="utf-8")
    json.dump(id2label, labels_file)

    json.dump(train_data, open('train_data.json', "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    label2id = {}
    for i, j in id2label.items():
        label2id[j] = i
    num_labels = len(label2id.keys()) * 2 + 1
    return id2label, label2id, num_labels


id2label, label2id, num_labels = get_id2label(label_path="bmes_train.rbtl.labels.json",
                                              train_path=wait_train_data)

# 建立分词器
tokenizer = Tokenizer(rbtl_dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < max_text_length:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = int(label2id[l]) * 2 + 1
                        I = int(label2id[l]) * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(
    rbtl_config_path,
    rbtl_checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans

        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('../bmes_models/' + str(self.best_val_f1) + 'bmes.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


def train_model():
    evaluator = Evaluator()
    train_data, test_data, valid_data, _ = load_data_tri(wait_train_data)

    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )


def predict_model():
    filename = '../bmes_models/0.6957148001440513bmes.weights'
    model.load_weights(filename)
    medical_dicts_drop_duplicates = json.load(open("../data/bmes_test.json", "r",
                                                   encoding="utf-8"))
    export = []

    for i in tqdm(medical_dicts_drop_duplicates):
        R = NER.recognize(i["text"])
        res = ['-'.join(x) for x in R]
        export.append({"id": i["id"], "entities": res})
    json.dump(export, open("entities.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    train_model()

    predict_model()
