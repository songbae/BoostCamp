import re
import os
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import numpy as np
import torchvision
import cv2
import sys
import albumentations as A
import pandas as pd


def prepare_data(data_path):
    data = {'path': [], 'label': []}
    for label in os.listdir(data_path):
        label_temp = None
        if label[:2] == '._':
            continue
        sub = os.path.join(data_path, label)
        for img in os.listdir(sub):
            if img[0] == '.':
                continue
            if img.find('normal') != -1:
                label_temp = 0
            elif img.find('incorrect') != -1:
                label_temp = 2
            else:
                label_temp = 1

            data['path'].append(os.path.join(sub, img))
            data['label'].append(label_temp)
    return pd.DataFrame(data)


def make_dataset(label_data, age_num):
    temp = {'gender': [], 'Age': []}
    train_df = pd.read_csv('/opt/ml/input/data/train/train.csv')
    temp1 = pd.DataFrame(np.digitize(train_df.iloc[:, 3].values, [
        29, age_num, 100], [0, 1, 2]), columns=['Age'])
    train_df = pd.concat([train_df, temp1], axis=1)
    train_df = train_df.drop(['age'], axis=1)
    train_df['gender'] = train_df['gender'].map({'female': 0, 'male': 1})
    train_df = train_df.drop(['race'], axis=1)
    for idx, i in enumerate(label_data.iloc[:, 0].values):
        classes = i.split('/')[-2]
        for row in train_df.values:
            if classes == row[2]:
                temp['gender'].append(row[1])
                temp['Age'].append(row[3])
    return pd.DataFrame(temp)


def concat_dataset(label_df, labels):

    label_df = pd.concat([label_df, labels['Age'], labels['gender']], axis=1)
    for idx, i in enumerate(label_df.values):
        temp = i[0]
        temp = re.sub('images', 'new_img', temp)
        label_df.iloc[idx, 0] = temp
    return label_df


class PstageDataset(Dataset):
    def __init__(self, path_data, transforms=None, class_num=None, idx=None, resize=224, age_num=None):
        self.class_num = class_num
        self.index = idx
        self.transforms = Customargumentation(mode=transforms, resize=resize)
        self.path_data = prepare_data(path_data)
        self.make_dataset = make_dataset(self.path_data, age_num=age_num)
        self.data = concat_dataset(
            self.path_data, self.make_dataset).iloc[self.index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index, 0]
        label = int(self.data.iloc[index, self.class_num])
        image = cv2.imread(img_path)
        assert image.shape[2] == 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image=image)
        image = image.transpose((2, 0, 1))
        sample = {'image': image, 'label': label}
        return sample


class Customargumentation:
    '''
    mask-> tfms
    age,gender-> tfms
    train_mode-> tfms
    valid_mode -> tfms
    '''

    def __init__(self, mode=None, resize=224):
        # assert type(resize) == list, f'resize type is not list '
        if mode == 'train_tfms_mask':
            self.transform = A.Compose([
                A.OneOf([
                    A.Perspective(p=1.0),
                    A.Rotate(limit=20, p=1.0, border_mode=1),
                ], p=0.5),
                A.OneOf([
                    A.RandomBrightness(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.RandomContrast(p=1.0),
                ], p=0.5),
                A.Compose([
                    A.Resize(resize, resize),
                    A.Normalize(),
                ])
            ])
        elif mode == 'train_age_gender':

            self.transform = A.Compose([
                A.Rotate(limit=20, p=0.5, border_mode=1),
                A.OneOf([
                    A.RandomGridShuffle(grid=(2, 2), p=1.0),# not using for gender
                    # A.RandomGridShuffle(grid=(4, 2), p=1.0),
                    A.Perspective(p=1.0)
                ], p=0.5),
                A.GaussNoise(p=0.5),
                A.Compose([
                    A.Resize(resize, resize),
                    A.Normalize(),
                ])
            ])
#        elif mode =
        elif mode == 'valid_tfms':
            self.transform = A.Compose([
                A.Resize(resize, resize),
                A.Normalize(),
            ])

    def __call__(self, image):
        return self.transform(image=image)['image']


class PstageDataset_test(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = Customargumentation(mode=transforms, resize=224)
        self.path = '/opt/ml/input/data/eval/new_images'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index, 0]
        image = cv2.imread(os.path.join(self.path, img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)
        image = image.transpose((2, 0, 1))
        sample = {'image': image}
        return sample


class tta_dataset(Dataset):
    def __init__(self, data, transforms=None, tta_tfms=None):
        self.data = data
        self.transform = Customargumentation(mode=transforms, resize=224)
        self.tta_tfms = Customargumentation(mode='valid_tfms')
        self.path = '/opt/ml/input/data/eval/new_images'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index, 0]
        image = cv2.imread(os.path.join(self.path, img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = [self.tta_tfms(image=image)]
        images.extend([
            self.transform(image=image) for _ in range(7)
        ])
        images = [
            img.transpose((2, 0, 1)) for img in images
        ]
        sample = {
            'image': images
        }
        return sample
