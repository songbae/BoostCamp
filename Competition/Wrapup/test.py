import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
import sys
import copy
import random
from efficientnet_pytorch import EfficientNet
from glob import glob
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from data.dataset import PstageDataset_test, Customargumentation, tta_dataset
import argparse
import time
from tqdm import tqdm


def make_submission_tta(args):

    classes = {
        '110': 0,
        '111': 1,
        '112': 2,
        '100': 3,
        '101': 4,
        '102': 5,
        '210': 6,
        '211': 7,
        '212': 8,
        '200': 9,
        '201': 10,
        '202': 11,
        '010': 12,
        '011': 13,
        '012': 14,
        '000': 15,
        '001': 16,
        '002': 17
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = pd.read_csv('/opt/ml/input/data/eval/info.csv')
    submission = pd.read_csv(os.path.join(
        '/opt/ml/input/data/eval', 'info.csv'))
    test_data_path = '/opt/ml/input/data/eval/new_images'
    model_mask = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_age = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_gender = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=2).to(device)
    model_mask.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_mask0.pth'))
    model_age.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_allage0.pth'))
    model_gender.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_allgender1.pth'))
    model_age.eval()
    model_mask.eval()
    model_gender.eval()
    #######################################################################
    tta_data = tta_dataset(test_data, transforms='train_age_gender')
    tta_dl = DataLoader(tta_data, batch_size=32, shuffle=False)
    test_dataset = PstageDataset_test(test_data, transforms='valid_tfms')
    test_dl = DataLoader(test_dataset, batch_size=32,
                         shuffle=False)
    total_answer = list()
    for sample, samples in tqdm(zip(test_dl, tta_dl)):
        with torch.no_grad():
            inputs = sample['image'].to(device)
            output_mask = model_mask(inputs)
            output_gender = model_gender(inputs)
            #output_age = model_age(inputs)
            output_age = 0
            for images in samples['image']:
                output_age += model_age(images.to(device))
            _, preds_age = torch.max(output_age, 1)
            _, preds_mask = torch.max(output_mask, 1)
            _, preds_gender = torch.max(output_gender, 1)
            for mask, gender, age in zip(preds_mask, preds_gender, preds_age):
                ans = list()
                ans.append(mask.detach().cpu().numpy().astype(
                    '|S1').tostring().decode('utf-8'))
                ans.append(gender.detach().cpu().numpy().astype(
                    '|S1').tostring().decode('utf-8'))
                ans.append(age.detach().cpu().numpy().astype(
                    '|S1').tostring().decode('utf-8'))
                data = ''.join(ans)
                total_answer.append(classes[data])
    submission['ans'] = total_answer
    submission.to_csv(f'{args.save_dir}{time.time()}.csv', index=False)
#######################################################


def make_submission(args):

    classes = {
        '110': 0,
        '111': 1,
        '112': 2,
        '100': 3,
        '101': 4,
        '102': 5,
        '210': 6,
        '211': 7,
        '212': 8,
        '200': 9,
        '201': 10,
        '202': 11,
        '010': 12,
        '011': 13,
        '012': 14,
        '000': 15,
        '001': 16,
        '002': 17
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = pd.read_csv('/opt/ml/input/data/eval/info.csv')
    submission = pd.read_csv(os.path.join(
        '/opt/ml/input/data/eval', 'info.csv'))
    test_data_path = '/opt/ml/input/data/eval/new_images'
    model_mask = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_age = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_gender = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=2).to(device)
    model_mask.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_mask0.pth'))
    model_age.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_allage0.pth'))
    model_gender.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_allgender1.pth'))

    model_age.eval()
    model_mask.eval()
    model_gender.eval()
    test_dataset = PstageDataset_test(test_data, transforms='valid_tfms')
    test_dl = DataLoader(test_dataset, batch_size=32,
                         shuffle=False)
    total_answer = list()
    for sample in tqdm(test_dl):
        with torch.no_grad():
            inputs = sample['image'].to(device)
            output_mask = model_mask(inputs)
            output_gender = model_gender(inputs)
            output_age = model_age(inputs)
            _, preds_age = torch.max(output_age, 1)
            _, preds_mask = torch.max(output_mask, 1)
            _, preds_gender = torch.max(output_gender, 1)
            for mask, gender, age in zip(preds_mask, preds_gender, preds_age):
                ans = list()
                ans.append(mask.detach().cpu().numpy().astype(
                    '|S1').tostring().decode('utf-8'))
                ans.append(gender.detach().cpu().numpy().astype(
                    '|S1').tostring().decode('utf-8'))
                ans.append(age.detach().cpu().numpy().astype(
                    '|S1').tostring().decode('utf-8'))
                data = ''.join(ans)
                total_answer.append(classes[data])
    submission['ans'] = total_answer
    submission.to_csv(f'{args.save_dir}{time.time()}.csv', index=False)


def make_submission1(args):

    classes = {
        '110': 0,
        '111': 1,
        '112': 2,
        '100': 3,
        '101': 4,
        '102': 5,
        '210': 6,
        '211': 7,
        '212': 8,
        '200': 9,
        '201': 10,
        '202': 11,
        '010': 12,
        '011': 13,
        '012': 14,
        '000': 15,
        '001': 16,
        '002': 17
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = pd.read_csv('/opt/ml/input/data/eval/info.csv')
    submission = pd.read_csv(os.path.join(
        '/opt/ml/input/data/eval', 'info.csv'))
    test_data_path = '/opt/ml/input/data/eval/new_images'
    model_mask = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_age = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_age1 = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_age2 = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_age3 = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_age4 = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=3).to(device)
    model_gender = EfficientNet.from_pretrained(
        'efficientnet-b3', num_classes=2).to(device)
    model_mask.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_mask0.pth'))
    model_age.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_age0.pth'))
    model_age1.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_age1.pth'))
    model_age2.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_age2.pth'))
    model_age3.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_age3.pth'))
    model_age4.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_age4.pth'))
    models = list()
    models.extend([model_age, model_age1, model_age2, model_age3, model_age4])
    model_gender.load_state_dict(torch.load(
        '/opt/ml/Pstage/log/model_save_gender0.pth'))

    model_mask.eval()
    model_gender.eval()
    test_dataset = PstageDataset_test(test_data, transforms='valid_tfms')
    test_dl = DataLoader(test_dataset, batch_size=32,
                         shuffle=False)
    total_answer = list()
    for sample in tqdm(test_dl):
        with torch.no_grad():
            inputs = sample['image'].to(device)
            output_mask = model_mask(inputs)
            output_gender = model_gender(inputs)
            output_age = 0
            for model in models:
                model.eval()
                output_age += model(inputs)
            _, preds_age = torch.max(output_age, 1)
            _, preds_mask = torch.max(output_mask, 1)
            _, preds_gender = torch.max(output_gender, 1)
            for mask, gender, age in zip(preds_mask, preds_gender, preds_age):
                ans = list()
                ans.append(mask.detach().cpu().numpy().astype(
                    '|S1').tostring().decode('utf-8'))
                ans.append(gender.detach().cpu().numpy().astype(
                    '|S1').tostring().decode('utf-8'))
                ans.append(age.detach().cpu().numpy().astype(
                    '|S1').tostring().decode('utf-8'))
                data = ''.join(ans)
                total_answer.append(classes[data])
    submission['ans'] = total_answer
    submission.to_csv(f'{args.save_dir}{time.time()}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    save_dir = '/opt/ml/Pstage/log/'
    parser.add_argument('--save_dir', type=str,
                        default=save_dir, help='default save_dir== log')
    parser.add_argument('--name', type=str, default='submission_')
    parser.add_argument('--tta', type=str, default='yes')

    args = parser.parse_args()
    print('test_start')
    print('*'*40)
    print(args)
    if args.tta == 'yes':
        make_submission_tta(args)
    elif args.tta == 'all':
        make_submission1(args)
    else:
        make_submission(args)
