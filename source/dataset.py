import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        features = self.dataset[index]['features']
        cst = np.concatenate((features['chroma'], features['spectral_contrast'], features['tonnetz']))
        if self.mode == 'LMC':
            # create the LMC feature
            # combine LM & CST
            lm = features['logmelspec']
            feature = np.concatenate((lm,cst))
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            # create the MC feature
            # combine MFCC & CST
            mfcc = features['mfcc']
            feature = np.concatenate((mfcc,cst))
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            # create the MLMC feature
            # combine MFCC, LMC, & CST
            lm = features['logmelspec']
            mfcc = features['mfcc']
            feature = np.concatenate((mfcc,lm,cst))
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)
