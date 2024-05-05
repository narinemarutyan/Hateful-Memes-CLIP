from typing import Optional, Dict, Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HatefulMemesDataset(Dataset):
    """
    A dataset class for loading and transforming the Hateful Memes dataset
    """

    def __init__(self, image_path: str, csv_path: str, split: str, image_size: Optional[int] = 224) -> None:
        """
        Initialize the Hateful Memes Dataset

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
        """
        super(HatefulMemesDataset, self).__init__()
        self.image_path = image_path
        self.split = split
        self.image_size = image_size
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns
        -------
        out : int
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
        sample['image'] = Image.open(f"{self.image_path}/{image_fn}").convert('RGB').resize(
            (self.image_size, self.image_size))
        sample['text'] = row['text']
        sample['label'] = row['label']
        sample['caption'] = row['caption']

        return sample
