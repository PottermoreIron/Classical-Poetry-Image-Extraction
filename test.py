from data_loader import PoetryImageDataSet
import numpy as np
import config
from torch.utils.data import DataLoader
from tqdm import tqdm

# data = np.load(config.train_dir, allow_pickle=True)
# word_train = data['words']
# label_train = data['labels']
#
# train_dataset = PoetryImageDataSet(word_train, label_train, config)
# train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
#                           shuffle=True, collate_fn=train_dataset.collate_fn)
#

