# 多线程字典最大正向匹配分词

import json
import threading
from copy import deepcopy
from queue import Queue

import pandas as pd


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


ner_label = json.load(open("../data/medical_entity/entity.json", "r", encoding="utf-8"))


def label_append_max(labels, label):
    """
    新增实体进行验证
    :param labels:实体集合
    :param label: 当前实体
    :return:
    """
    if isinstance(label, int):
        return
    if len(labels) == 0:
        labels.extend(label)
        for i in label:
            for label in (merge_list(labels, i)):
                if label not in labels:
                    labels.append(label)

        return

    elif len(labels) > 0:
        for label in label:
            for label in (merge_list(labels, label)):
                if label not in labels:
                    labels.append(label)


def merge_list(labels, label):
    second_labels = deepcopy(labels)
    for in_label in range(len(labels)):
        if isinstance(label, list) and label not in labels:
            if labels[in_label][0] <= label[0] <= labels[in_label][1] and label[1] - label[0] > labels[in_label][1] - \
                    labels[in_label][0]:
                if labels[in_label] in second_labels:
                    second_labels.remove(labels[in_label])
            elif labels[in_label][0] <= label[1] <= labels[in_label][1] and label[1] - label[0] > labels[in_label][1] - \
                    labels[in_label][0]:
                if labels[in_label] in second_labels:
                    second_labels.remove(labels[in_label])
    labels = second_labels
    second_labels = deepcopy(labels)

    for in_label in range(len(labels)):
        if label not in labels:
            if labels[in_label][0] <= label[0] <= labels[in_label][1] and label[1] - label[0] > labels[in_label][1] - \
                    labels[in_label][0]:
                second_labels.append(label)
            elif labels[in_label][0] <= label[1] <= labels[in_label][1] and label[1] - label[0] > labels[in_label][1] - \
                    labels[in_label][0]:
                second_labels.append(label)
            elif not labels[in_label][0] <= label[0] <= labels[in_label][1] and not labels[in_label][0] <= label[
                1] <= labels[in_label][1]:
                second_labels.append(label)
    return second_labels


def medical_ner():
    if isinstance(medical_questions, list):
        for medical_question in medical_questions:
            medical_question_ner(medical_question)


def medical_question_ner(medical_question=None):
    if isinstance(medical_question, list):
        for medical_message in medical_question:
            if isinstance(medical_message, str):

                medical_message = medical_message.strip()
                medical_message = medical_message.replace("\n", "")
                medical_message = medical_message.replace("\r", "")
                medical_message = medical_message.replace(" ", "")
            else:
                continue
            label_index_list = []
            for i, j in ner_label.items():
                if i in medical_message:
                    label_index = index_of_str(medical_message, i, j)
                    if len(label_index) >= 1:
                        label_append_max(label_index_list, label_index)

            if len(label_index_list) > 2 and len(medical_message) > 10:
                entity_dict_label = {"text": medical_message, "labels": label_index_list}
                entity_dict_label = json.dumps(entity_dict_label, ensure_ascii=False)

                fp.write(entity_dict_label + "\n")


class BSSpider(threading.Thread):
    def __init__(self, page_queue, *args, **kwargs):
        super(BSSpider, self).__init__(*args, **kwargs)
        self.page_queue = page_queue

    def run(self):
        while True:
            if self.page_queue.empty():
                break
            try:
                medical_question = self.page_queue.get()
                if isinstance(medical_question, list):
                    for medical_message in medical_question:
                        if isinstance(medical_message, str):

                            medical_message = medical_message.strip()
                            medical_message = medical_message.replace("\n", "")
                            medical_message = medical_message.replace("\r", "")
                            medical_message = medical_message.replace(" ", "")
                        else:
                            continue
                        label_index_list = []
                        for i, j in ner_label.items():
                            if len(i) > 3 and i in medical_message:

                                label_index = index_of_str(medical_message, i, j)
                                if len(label_index) >= 1:
                                    label_append_max(label_index_list, label_index)

                        if len(label_index_list) > 2 and len(medical_message) > 10:
                            label_index_list = sorted(label_index_list, key=lambda k: k[0], reverse=False)
                            export_list = [label_index_list[0]]
                            for label_index in range(1, len(label_index_list)):
                                now_label = label_index_list[label_index]
                                last_label = label_index_list[label_index - 1]
                                if last_label[1] >= now_label[0]:
                                    if last_label[1] - last_label[0] >= now_label[1] - now_label[0]:
                                        continue
                                    elif last_label in export_list:

                                        export_list.remove(last_label)
                                        export_list.append(now_label)
                                    else:
                                        continue
                                else:
                                    export_list.append(now_label)
                            entity_dict_label = {"text": medical_message, "labels": export_list}
                            entity_dict_label = json.dumps(entity_dict_label, ensure_ascii=False)

                            fp.write(entity_dict_label + "\n")
            except Exception as e:
                print(e)


if __name__ == '__main__':
    medical_questions = pd.read_csv("../data/breath/breath.answer.csv").drop_duplicates().values.tolist()
    fp = open('../data/breath/breath_answer_entity.json', 'w', newline='', encoding='utf-8')

    page_queue = Queue()

    for url in medical_questions:
        page_queue.put(url)

    for x in range(8):
        t = BSSpider(page_queue)
        t.start()
