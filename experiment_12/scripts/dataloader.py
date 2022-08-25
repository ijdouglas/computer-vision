import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
import os.path as osp
from PIL import Image
import random
from torchvision.datasets.folder import default_loader
import collections


class Egocentric(data_utils.Dataset):
    def __init__(self, csv=None, root_dir=None, target_number=1, label_key = 'joint_attention',
                  seed=1, train=True, transform=None, loader=default_loader):
        self.csv = csv
        self.df = pd.read_csv(self.csv)
        self.root_dir = root_dir
        self.train = train
        self.num_in_train = len(self.df)
        # self.num_in_train = 1000
        self.num_in_test = 100
        self.r = np.random.RandomState(seed)
        self.target_number = target_number
        self.transform = transform
        self.loader = default_loader
        self.img_key = 'path'
        self.label_key = label_key # column name of response variable/label
        label_freq = collections.Counter([self.get_label_idx(i) for i in range(len(self))])
        self.inverse_label_freq = [1.0*label_freq[i]/len(self) for i in range(len(label_freq))]

    def get_img_path(self, i):
        '''
        get img_path of i-th data point
        '''
        return os.path.join(self.root_dir, str(self.df.iloc[i][self.img_key]))

    def get_img(self, i):
        '''
        get img array of i-th data point
        self.transform is applied if exists
        '''
        img = self.loader(self.get_img_path(i))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_label(self,i):
        '''
        get label of i-th data point as it is. 
        '''
        return int(self.df.iloc[i][self.label_key])

    def get_label_idx(self,i):
        '''
        get label idx, which start from 0 incrementally
        self.target_transform is applied if exists
        '''
        label = self.get_label(i)
        # if self.target_transform is not None:
        #     if  isinstance(self.target_transform, dict):
        #         label_idx = self.target_transform[label]
        #     else:
        #         label_idx = self.target_transform(label)
        # else:
        label_idx = int(label)
        return label_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        '''
        get (img,label_idx) pair of i-th data point
        img is already preprocessed
        label_idx start from 0 incrementally 
        That is, they can be used for cnn input directly
        '''
        return {"input":self.get_img(i), "label":self.get_label_idx(i), "img_path": self.get_img_path(i)}

class EgocentricSequence(data_utils.Dataset):
    def __init__(self, csv=None, root_dir=None, target_number=1, label_key='joint_attention', 
                  seed=1, train=True, transform=None, loader=default_loader):
        self.csv = csv
        self.df = pd.read_csv(self.csv)
        self.events = self.df.event_id.unique()
        self.root_dir = root_dir
        self.train = train
        self.num_in_train = len(self.df)
        # self.num_in_train = 1000
        self.num_in_test = 100
        self.r = np.random.RandomState(seed)
        self.target_number = target_number
        self.transform = transform
        self.loader = default_loader
        self.img_key = 'path'
        self.label_key = label_key # column name of response variable/label
        label_freq = collections.Counter([self.get_label_idx(i) for i in range(len(self))])
        self.inverse_label_freq = [1.0*label_freq[i]/len(self) for i in range(len(label_freq))]

    def get_img_path(self, i):
        '''
        get img_path of i-th data point
        '''
        return os.path.join(self.root_dir, str(self.df.iloc[i][self.img_key]))

    def get_img(self, i):
        '''
        get img array of i-th data point
        self.transform is applied if exists
        '''
        img = self.loader(self.get_img_path(i))
        if self.transform is not None:
            img = self.transform(img)
        return img


    def get_event(self, i):
        e = self.events[i]
        event_df = self.df.query("event_id == {}".format(e))

        return event_df


    def get_label(self,i):
        '''
        get label of i-th data point as it is. 
        '''
        return int(self.df[self.df.event_id == self.events[i]].iloc[0][self.label_key])
        # return int(self.df.iloc[i][self.label_key])

    def get_label_idx(self,i):
        '''
        get label idx, which start from 0 incrementally
        self.target_transform is applied if exists
        '''
        label = self.get_label(i)
        # if self.target_transform is not None:
        #     if  isinstance(self.target_transform, dict):
        #         label_idx = self.target_transform[label]
        #     else:
        #         label_idx = self.target_transform(label)
        # else:
        label_idx = int(label)
        return label_idx

    def __len__(self):
        return len(self.events)

    def __getitem__(self, i):
        '''
        get (img,label_idx) pair of i-th data point
        img is already preprocessed
        label_idx start from 0 incrementally 
        That is, they can be used for cnn input directly
        '''
        event = self.get_event(i)
        imgs = []
        
        for idx, row in event.iterrows():
            img_path = os.path.join(self.root_dir, str(row[self.img_key]))
            img = self.loader(img_path)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
            
        event = torch.stack(imgs) if len(imgs) < 5 else torch.stack(imgs[:5])
        
        return {
            "input": event, 
            "label": self.get_label_idx(i), 
            "img_path": self.get_img_path(i)
            }

def setup_dataloader(args):
    if args.normalization == 'mnist':
        norm_means = (0.1307,)
        norm_stds = (0.3081,)
    elif args.normalization == 'imagenet':
        norm_means = [0.485, 0.456, 0.406]
        norm_stds = [0.229, 0.224, 0.225]
    elif args.normalization == 'none':
        norm_means = (0,)
        norm_stds = (1,)
    train = Egocentric(csv=args.train_set,
                target_number=args.target_number,
                root_dir=args.root_dir,
                transform=transforms.Compose([
                    transforms.Resize((256, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_means, norm_stds)]),
                seed=args.seed,
                train=True,
                label_key = args.label_key)

    test = Egocentric(csv=args.test_set,
                    target_number=args.target_number,
                    root_dir=args.root_dir,
                    transform=transforms.Compose([
                        transforms.Resize((256, 384)),
                        transforms.ToTensor(),
                        transforms.Normalize(norm_means,norm_stds)]),
                    seed=args.seed,
                    train=True,
                    label_key = args.label_key)

    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = data_utils.DataLoader(train, batch_size=args.batch,
                                     shuffle=True,
                                     **loader_kwargs)

    test_loader = data_utils.DataLoader(test, batch_size=args.batch,
                                        shuffle=True,
                                        **loader_kwargs)

    return train_loader, test_loader, train, test

def setup_dataloader_seq(args):
    if args.normalization == 'mnist':
        norm_means = (0.1307,)
        norm_stds = (0.3081,)
    elif args.normalization == 'imagenet':
        norm_means = [0.485, 0.456, 0.406]
        norm_stds = [0.229, 0.224, 0.225]
    elif args.normalization == 'none':
        norm_means = (0,)
        norm_stds = (1,)
    train = EgocentricSequence(csv=args.train_set,
                target_number=args.target_number,
                root_dir=args.root_dir,
                transform=transforms.Compose([
                    transforms.Resize((256, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_means, norm_stds)]),
                seed=args.seed,
                train=True,
                label_key = args.label_key)

    test = EgocentricSequence(csv=args.test_set,
                    target_number=args.target_number,
                    root_dir=args.root_dir,
                    transform=transforms.Compose([
                        transforms.Resize((256, 384)),
                        transforms.ToTensor(),
                        transforms.Normalize(norm_means, nrom_stds)]),
                    seed=args.seed,
                    train=True,
                    label_key=args.label_key)

    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = data_utils.DataLoader(train, batch_size=args.batch,
                                     shuffle=True,
                                     **loader_kwargs)

    test_loader = data_utils.DataLoader(test, batch_size=args.batch,
                                        shuffle=True,
                                        **loader_kwargs)

    return train_loader, test_loader, train, test
