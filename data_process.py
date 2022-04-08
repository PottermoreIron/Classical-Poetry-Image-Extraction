# process the data
import os
import json
import numpy as np
import logging


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config

    def process(self):
        for file_name in self.config.files:
            self.preprocess(file_name)

    def preprocess(self, mode):
        input_dir = self.data_dir + mode + '.json'
        output_dir = self.data_dir + mode + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
            # 先读取到内存中，然后逐行处理
            for poem in original_data:
                text = poem.get('text', None)
                words = list(text)
                # 如果没有label，则返回None
                label_entities = poem.get('label', None)
                labels = ['O'] * len(words)

                if label_entities is not None:
                    for label_index in label_entities:
                        start_index, end_index = label_index[0], label_index[1]
                        if start_index+1 == end_index:
                            labels[start_index] = 'S-image'
                        else:
                            labels[start_index] = 'B-image'
                            labels[start_index + 1:end_index - 1] = ['I-image'] * (end_index - start_index - 2)
                            labels[end_index - 1] = 'E-image'
                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("--------train data process DONE!--------")
