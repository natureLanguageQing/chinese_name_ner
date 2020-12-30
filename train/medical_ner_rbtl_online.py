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

rbtl_config_path = '../chinese_rktl3/bert_config_rbtl3.json'
rbtl_checkpoint_path = '../chinese_rktl3/bert_model.ckpt'
rbtl_dict_path = '../chinese_rktl3/vocab.txt'
# rbtl_config_path = '../pre_train_language_model/rbtl3/bert_config_rbtl3.json'
# rbtl_checkpoint_path = '../pre_train_language_model/rbtl3/bert_model.ckpt'
# rbtl_dict_path = '../pre_train_language_model/rbtl3/vocab.txt'
wait_train_data = '../data/medical_ner/clean_medical_ner_entities_1229.json'


def index_of_str(s1, s2, label):
    s2 = s2.rstrip("\n")
    dex = 0
    index = []
    lt = s1.split(s2)
    num = len(lt)
    for i in range(num - 1):
        dex += len(lt[i])
        index.append([dex, len(s2) + dex, label, s2])
        dex += len(s2)
    return index


def load_data(filename):
    D = []
    labels = []
    f = open(filename, encoding='utf-8')
    medical = json.load(f)
    for medical in medical:
        d = []

        medical_text = medical["text"]
        medical_labels = medical["annotations"]
        next_label = 0

        for label_index in medical_labels:
            d.append([medical_text[next_label: label_index["start_offset"]], "O"])
            d.append([medical_text[label_index["start_offset"]: label_index["end_offset"]], label_index["label"]])
            next_label = label_index["end_offset"]
            labels.append(label_index["label"])
        D.append(d)

    labels = list(set(labels))
    return D, labels


load_data(wait_train_data)


def load_data_tri(filename):
    D = []
    labels = []
    f = open(filename, encoding='utf-8')
    medical = json.load(f)
    for medical in medical:
        d = []

        medical_text = medical["text"]
        medical_labels = medical["annotations"]
        next_label = 0

        for label_index in medical_labels:
            d.append([medical_text[next_label: label_index["start_offset"]], "O"])
            d.append([medical_text[label_index["start_offset"]: label_index["end_offset"]], label_index["label"]])
            next_label = label_index["end_offset"]
            labels.append(label_index["label"])
        D.append(d)
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
    label2id = {}
    for i, j in id2label.items():
        label2id[j] = i
    num_labels = len(label2id.keys()) * 2 + 1
    return id2label, label2id, num_labels


id2label, label2id, num_labels = get_id2label(label_path="bmes_train.rbtl.labels.json",
                                              train_path=wait_train_data)
max_text_length = 32
epochs = 10
batch_size = 16
bert_layers = 3
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

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
            model.save_weights('../medical_ner/' + str(self.best_val_f1) + 'medical_ner.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__ == '__main__':
    evaluator = Evaluator()
    train_data, test_data, valid_data, _ = load_data_tri(wait_train_data)

    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
