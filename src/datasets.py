import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HatefulMemesDataset(Dataset):
    """
    A dataset class for loading and transforming the Hateful Memes dataset
    """
    def __init__(self, root_folder, image_folder, split='train', image_size=224):
        """
        Initializes the Hateful Memes Dataset

        Parameters
        ----------
        image_path : str
            The directory where images are stored
        csv_path : str
            The path to the CSV file containing the dataset's metadata
        split : str
            The dataset split like train, val or test
        image_size : Optional[int] = 224
            The size to which the images should be resized

        Returns
        -------
        out : None
        """
        super(HatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.split = split
        self.image_size = image_size
        self.info_file = os.path.join(root_folder, 'hateful_memes_expanded_ofa.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)

    def __len__(self):
        """
        Returns the number of items in the dataset

        Returns
        -------
        out : int
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves the dataset item at the specified index

        Parameters
        ----------
        idx : int
            The index of the item to retrieve

        Returns
        -------
        out : Dict[str, Any]
        """
        row = self.df.iloc[idx]
        sample = {}
        image_fn = row['img'].split('/')[1]
        sample['image'] = Image.open(f"{self.image_folder}/{image_fn}").convert('RGB').resize(
            (self.image_size, self.image_size))
        sample['text'] = row['text']
        sample['label'] = row['label']
        sample['caption'] = row['caption']

        return sample
