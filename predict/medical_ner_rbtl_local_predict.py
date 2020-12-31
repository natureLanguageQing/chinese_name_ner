#! -*- coding: utf-8 -*-

import json

from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

from bert4keras.backend import K
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.tokenizers import Tokenizer

rbtl_config_path = '../pre_train_language_model/rbtl3/bert_config_rbtl3.json'
rbtl_checkpoint_path = '../pre_train_language_model/rbtl3/bert_model.ckpt'
rbtl_dict_path = '../pre_train_language_model/rbtl3/vocab.txt'
wait_predict_data = '../data/breath/breath.answer.csv'
model_path = "../medical_ner/0.8512227442476828medical_ner.weights"


def get_id2label(label_path):
    labels = json.load(open(label_path, "r", encoding="utf-8"))
    print(labels)
    label2id = {}
    id2label = dict(enumerate(labels))

    for i, j in id2label.items():
        label2id[j] = i
    num_labels = len(label2id.keys()) * 2 + 1
    return id2label, label2id, num_labels


id2label, label2id, num_labels = get_id2label(label_path="medical_train.ner.labels.json")
max_text_length = 128
batch_size = 16
bert_layers = 3
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# 建立分词器
tokenizer = Tokenizer(rbtl_dict_path, do_lower_case=True)

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
        """

        :param text:
        :return:
        """
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

if __name__ == '__main__':
    model.load_weights(model_path)
    medical_dicts_drop_duplicates = open(wait_predict_data, "r",
                                         encoding="utf-8")
    export = []

    for i in tqdm(medical_dicts_drop_duplicates.readlines()[1:1000]):
        R = NER.recognize(i.strip("\n")[:256])
        export.append({"text": i, "entities": R})
        print({"text": i, "entities": R})
    json.dump(export, open("breath.answer.json", "w", encoding="utf-8"), ensure_ascii=False)
