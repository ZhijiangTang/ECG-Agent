
import os

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import set_seed

class BaseFinetuneDataset(Dataset):
    def __init__(self,args=None, is_train=True,**kwargs):
        super().__init__()
        self.args = args
        self.is_train = is_train
        self.num_class = None

        self.train_X,self.train_Y,self.test_X,self.test_Y ,self.num_train = self.load_dataset()

    def __getitem__(self, index):
        if self.is_train:
            return self.train_X[index,:],self.train_Y[index,...]
        else:
            return self.test_X[index,:],self.test_Y[index,...]

    def deal_data(self,X,Y):
        self.num_class = len(set(Y.squeeze().numpy().ravel()))
        return X,Y

    def load_dataset(self):
        X = torch.load(os.path.join(self.args['dataset_path'],self.args['dataset_name'],f"{self.args['dataset_save']}X.pt"),weights_only=False)
        Y = torch.load(os.path.join(self.args['dataset_path'],self.args['dataset_name'],f"{self.args['dataset_save']}Y.pt"),weights_only=False)

        X,Y = self.deal_data(X,Y)
        set_seed(self.args['seed'])
        indices = torch.randperm(X.shape[0])[:int(X.shape[0]*self.args['sub_dataset'])]
        X = X[indices,:]
        Y = Y[indices]
        num_train = int(X.shape[0]*self.args['finetune_rate'])

        train_X = X[:num_train,:]
        train_Y = Y[:num_train]

        test_X = X[num_train:,:]
        test_Y = Y[num_train:]

        return train_X,train_Y,test_X,test_Y, num_train
    
    def __len__(self):
        if self.is_train:
            return len(self.train_X)
        else:
            return len(self.test_X)


class ClassificationFinetuneDataset(BaseFinetuneDataset):
    def __init__(self, args, is_train=True, **kwargs):
        if (args['dataset_balance'] == 'undersample') & ('Undersample_' not in args['dataset_save']):
            args['dataset_save'] = f"Undersample_{args['dataset_save']}"
        super().__init__(args, is_train)
        self.train_X, self.train_Y, self.test_X,self.test_Y = self.train_X.float(), self.train_Y.long(), self.test_X.float(),self.test_Y.long()
        if self.args['dataset_balance'] == 'oversample':
            self.train_X, self.train_Y = self.balance_data()

    def balance_data(self):
        classes, counts = torch.unique(self.train_Y, return_counts=True)
        maxnum_count = counts.max()

        train_X = []
        train_Y = []
        for i,c in enumerate(classes):
            count = counts[i]
            if count == maxnum_count:
                train_X.append(torch.concat([self.train_X[self.train_Y==c,:]],dim=0))
                train_Y.append(torch.full([train_X[-1].shape[0]], c))
                continue
            c_train_X = self.train_X[self.train_Y==c,:]
            weights = torch.ones(count, dtype=torch.float32)
            samples = torch.multinomial(weights, num_samples=(maxnum_count-count), replacement=True)
            train_X.append(torch.concat([c_train_X,c_train_X[samples]],dim=0))
            train_Y.append(torch.full([train_X[-1].shape[0]], c))

        train_X = torch.concat(train_X,dim=0)
        train_Y = torch.concat(train_Y,dim=0)
        return train_X,train_Y


class DetectionFinetuneDataset(BaseFinetuneDataset):
    def __init__(self, args=None, is_train=True, **kwargs):
        super().__init__(args, is_train)
        self.train_X, self.train_Y, self.test_X,self.test_Y = self.train_X.float(), self.train_Y.float(), self.test_X.float(),self.test_Y.float()

    def deal_data(self, X, Y):
        patch_len = self.args['patch_len']
        _, T = X.shape
        if T % patch_len != 0:
            X = X[:,:int(patch_len*(T//patch_len))]
            Y = Y[:,:int(patch_len*(T//patch_len))]

        return X,Y

class ForecastFinetuneDataset(BaseFinetuneDataset):
    def __init__(self, args=None, is_train=True, **kwargs):
        super().__init__(args, is_train)
        self.all_step = None
        self.train_X, self.train_Y, self.test_X,self.test_Y = self.train_X.float(), self.train_Y.float(), self.test_X.float(),self.test_Y.float()

    def deal_data(self, X, Y):
        patch_len = self.args['patch_len']
        _, T = X.shape
        if T % patch_len != 0:
            _,Y_T = Y.shape
            Y = torch.concat([X[:,int(patch_len*(T//patch_len)):],Y],dim=-1)
            Y = Y[:,:int(patch_len*(Y_T//patch_len))]
            X = X[:,:int(patch_len*(T//patch_len))]
        self.all_step = Y.shape[1]//patch_len

        return X,Y


class GenerationFinetuneDataset(DetectionFinetuneDataset):
    def __init__(self, args=None, is_train=True, **kwargs):
        super().__init__(args, is_train)
    