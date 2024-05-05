import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HatefulMemesDataset(Dataset):
    def __init__(self, root_folder, image_folder, split='train', labels='original', image_size=224):
        super(HatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.split = split
        self.labels = labels
        self.image_size = image_size
        self.info_file = os.path.join(root_folder, 'hateful_memes_expanded.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

        if split in ['test_seen', 'test_unseen']:
            self.fine_grained_labels = []
        elif self.labels == 'fine_grained':
            self.pc_columns = [col for col in self.df.columns if col.endswith('_pc') and not col.endswith('_gold_pc')]
            self.pc_columns.remove('gold_pc')
            self.attack_columns = [col for col in self.df.columns if
                                   col.endswith('_attack') and not col.endswith('_gold_attack')]
            self.attack_columns.remove('gold_attack')
            self.fine_grained_labels = self.pc_columns + self.attack_columns
        elif self.labels == 'fine_grained_gold':
            self.pc_columns = [col for col in self.df.columns if col.endswith('_gold_pc')]
            self.attack_columns = [col for col in self.df.columns if col.endswith('_gold_attack')]
            self.fine_grained_labels = self.pc_columns + self.attack_columns
        else:
            self.fine_grained_labels = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        image_fn = row['img'].split('/')[1]
        item['image'] = Image.open(f"{self.image_folder}/{image_fn}").convert('RGB').resize(
            (self.image_size, self.image_size))
        item['text'] = row['text']
        item['label'] = row['label']
        item['idx_meme'] = row['id']
        item['idx_image'] = row['pseudo_img_idx']
        item['idx_text'] = row['pseudo_text_idx']
        item['caption'] = row['caption']

        if self.labels.startswith('fine_grained'):
            for label in self.fine_grained_labels:
                item[label] = row[label]

        return item

