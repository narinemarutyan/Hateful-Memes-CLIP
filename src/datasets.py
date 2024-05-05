import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HatefulMemesDataset(Dataset):
    def __init__(self, image_folder, csv_folder, split, image_size=224):
        super(HatefulMemesDataset, self).__init__()
        self.image_folder = image_folder
        self.split = split
        self.image_size = image_size
        self.df = pd.read_csv(csv_folder)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {}
        image_fn = row['img'].split('/')[1]
        sample['image'] = Image.open(f"{self.image_folder}/{image_fn}").convert('RGB').resize(
            (self.image_size, self.image_size))
        sample['text'] = row['text']
        sample['label'] = row['label']
        sample['caption'] = row['caption']

        return sample
