import tqdm
import madgrad  # madgrad optimzer
import re
import json
import argparse
from torch.utils.tensorboard import SummaryWriter
from loss import FocalLoss
from sklearn.metrics import f1_score
from pathlib import Path
from importlib import import_module
from torch.utils.data import RandomSampler
from data.dataset import *
from sklearn.model_selection import KFold
import albumentations as A
import cv2
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
import random
import copy
import sys
import os
import time
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
sys.path.append('..')


def seed_everything(seed):
    '''
    seed 값 고정하기
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)


def get_idx(agrs):
    '''
    사람별 인덱스 나누기 kfold 5
    '''
    label_index = [i for i in range(2700)]
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_num, valid_num = list(), list()
    for idx, (train_idx, valid_idx) in enumerate(kfold.split(label_index)):
        train_idx = train_idx * 7
        valid_idx = valid_idx * 7
        train_temp, valid_temp = list(), list()
        for i in train_idx:
            for j in range(7):
                train_temp.append(i + j)
        for i in valid_idx:
            for j in range(7):
                valid_temp.append(i + j)
        train_num.append(train_temp)
        valid_num.append(valid_temp)
    return train_num, valid_num


def get_idx_label(args):
    label_index = [i for i in range(2700 * 7)]
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_num, valid_num = list(), list()
    for idx, (train_idx, valid_idx) in enumerate(kfold.split(label_index)):
        train_num.append(train_idx)
        valid_num.append(valid_idx)
    return train_num, valid_num


def train(data_dir, args):
    seed_everything(args.seed)
    # logger
    logger = SummaryWriter(log_dir=args.save_dir)
    with open(os.path.join(args.save_dir, f'config{args.test_method}.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    best_f1 = 0.0
    train_number, valid_number = get_idx_label(args)
    assert len(train_number) == 5, f'{train_number}'
    #train_number, valid_number = get_idx(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.loss == "crossentropy_loss":
        criterion = nn.CrossEntropyLoss()  # 이부분의 loss를 자유롭게 바꿀수 있는 코드로 바꾸기
    elif args.loss == 'focal_loss':
        criterion = FocalLoss()
    model = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=args.class_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_data = PstageDataset(data_dir, class_num=args.model_num,
                               idx=[i for i in range(2700*7)], transforms=args.train_transform, resize=args.resize, age_num=args.age_num)
    valid_data = PstageDataset(data_dir, class_num=args.model_num,
                               idx=valid_number[args.kfold_use], transforms='valid_tfms', resize=args.resize, age_num=args.age_num)
    train_sampler = RandomSampler(train_data)
    valid_sampler = RandomSampler(valid_data)
    train_dl = DataLoader(train_data, batch_size=args.batch_size,
                          shuffle=False, sampler=train_sampler, num_workers=4)
    valid_dl = DataLoader(valid_data, batch_size=args.batch_size,
                          shuffle=False, sampler=valid_sampler, num_workers=4)
    train_loss, valid_loss, train_acc, valid_acc = list(), list(), list(), list()
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}/{args.epochs-1}')
        print('-' * 20)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss, running_corrects, f1, f1_sc, num_cnt = 0.0, 0.0, 0.0, 0.0, 0.0
            if phase == 'train':
                for sample in tqdm.tqdm(train_dl):
                    inputs, label = sample['image'], sample['label']
                    inputs = inputs.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, label)
                        loss.backward()
                        optimizer.step()
                        num_cnt += args.batch_size
                        running_corrects += torch.sum(preds == label.data)
                        running_loss += loss.item()
                        f1_sc += f1_score(label.detach().cpu(),
                                          preds.detach().cpu(), average=args.average)

            if phase == 'valid':
                for sample in tqdm.tqdm(valid_dl):
                    inputs, label = sample['image'], sample['label']
                    inputs = inputs.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, label)
                        f1_sc += f1_score(label.detach().cpu(),
                                          preds.detach().cpu(), average=args.average)
                        running_corrects += torch.sum(
                            preds == label.data)
                        running_loss += loss.item()
                        num_cnt += args.batch_size

            # if phase == 'train':
            #     scheduler(running_loss)
            f1 = f1_sc/(num_cnt/args.batch_size)
            epoch_loss = float(running_loss)
            epoch_acc = float(
                running_corrects / num_cnt*100)
            logger.add_scalar(f'{phase}_loss', epoch_loss, epoch)
            logger.add_scalar(f'{phase}_acc', epoch_acc, epoch)
            logger.add_scalar(f'{phase}_f1_score', f1, epoch)

            print(
                f'{phase}Loss_{args.model_name}:{epoch_loss:.4f} Acc_{args.model_name}{epoch_acc:.2f} F1_score{f1:.4f}')

            if phase == 'valid' and f1 >= best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model.state_dict())
                model_path = f'/opt/ml/Pstage/log/model_save_all{args.model_name}{args.kfold_use}.pth'
                torch.save(best_model, model_path)
                print('best_model saved')
    return model.state_dict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data and model checkpoints directories
    parser.add_argument('--kfold_use', type=int, default=0,
                        help=' if use 2kold change to 1')
    parser.add_argument('--test_method', type=str, default="1",
                        help='making config file of which parameter used')
    parser.add_argument('--resize', type=int, default=224,
                        help='resizeing the imge defalut will be (224,224)')
    parser.add_argument('--seed', type=int, default=777,
                        help='default seed= 777 you can change any number ')
    parser.add_argument('--average', type=str, default='macro',
                        help='default macro( more than 2 classify, but if in gender you have to use binary ')
    parser.add_argument('--train_transform', type=str,
                        default='train_tfms_mask')
    parser.add_argument('--class_num', type=int, default=3,
                        help='model to classify defalut==3  mask=3, age=3, gender=2')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epoch defalut == 30')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='default batch_size==64')
    parser.add_argument('--model_num', type=int, default=1,
                        help='which label to train defalut =1 mask 2=age 3 = gender')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help=' optimizer tpye Adam,AdamW, Sgd, madgrad')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate defalut=3e-4')
    parser.add_argument('--loss', type=str, default='crossentropy_loss',
                        help='default crossentropy_loss , list-> focal_loss ( in age model will be best)')
    parser.add_argument('--log_interval', type=int, default=40,
                        help=' how many batchs to wait before logging')
    parser.add_argument('--model_name', type=str, default='mask',
                        help='model save at save/{model_name}')
    parser.add_argument('--save_dir', type=str, default='/opt/ml/Pstage/log/')
    parser.add_argument('--age_num', type=int, default=58, help='60 age ')
    args = parser.parse_args()

    print(args)

    data_dir = '/opt/ml/input/data/train/images'
    print('starting..')
    print('*'*20)
    train(data_dir, args)
