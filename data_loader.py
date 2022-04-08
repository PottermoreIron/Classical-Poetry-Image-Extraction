# loader for data
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
import torch


class PoetryImageDataSet(Dataset):
    def __init__(self, words, labels, config, word_pad_elem=0, label_pad_elem=-1):
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=True)
        self.label2id = config.label2id
        self.id2label = config.id2label
        self.dataset = self.data2idx(words, labels)
        self.word_pad_elem = word_pad_elem
        self.label_pad_elem = label_pad_elem
        self.device = config.device

    def data2idx(self, original_poems, original_labels):
        data = []
        poems = []
        labels = []
        for poem in original_poems:
            words = []
            word_lens = []
            for token in poem:
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            # 变成单个字的列表，开头加上[CLS]
            words = ['[CLS]'] + [item for token in words for item in token]
            token_start_ids = 1 + np.cumsum([0] + word_lens[:-1])
            poems.append((self.tokenizer.convert_tokens_to_ids(words), token_start_ids))
        for poem_labels in original_labels:
            poem_label_id = [self.label2id.get(label) for label in poem_labels]
            labels.append(poem_label_id)
        for poem, label in zip(poems, labels):
            data.append((poem, label))
        return data

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        poems = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        batch_len = len(poems)

        max_poem_len = max([len(p[0]) for p in poems])
        max_label_len = 0

        batch_data = self.word_pad_elem * np.ones((batch_len, max_poem_len))
        batch_label_starts = []

        # padding and aligning
        for i in range(batch_len):
            cur_len = len(poems[i][0])
            batch_data[i][:cur_len] = poems[i][0]
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = poems[i][-1]
            label_starts = np.zeros(max_poem_len)
            label_starts[[idx for idx in label_start_idx if idx < max_poem_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = self.label_pad_elem * np.ones((batch_len, max_label_len))
        for i in range(batch_len):
            cur_tags_len = len(labels[i])
            batch_labels[i][:cur_tags_len] = labels[i]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_label_starts, batch_labels]
