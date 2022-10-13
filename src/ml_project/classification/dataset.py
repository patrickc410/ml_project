import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

# import pandas as pd
import numpy as np

# from ml_project.preprocess import features


class MusicDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self):
        df_len = self.df.shape[0]
        return df_len

    def __getitem__(self, index: int):

        return torch.tensor(self.X[index, :])


def get_dataloader(dataset, batch_size: int = 16):
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
