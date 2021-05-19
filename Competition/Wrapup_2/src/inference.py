# Suppress warnings
from itertools import combinations
import random
import gc
import sys
import os
from features import generate_label, feature_engineering1
from utils import seed_everything, print_score,train_dataset,test_dataset,Model
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import argparse
import dateutil.relativedelta
import datetime
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from model import *
warnings.filterwarnings('ignore')


# Machine learning


# Custom library


TOTAL_THRES = 300  # 구매액 임계값
SEED = 777  # 랜덤 시드
seed_everything(SEED)  # 시드 고정


data_dir = '../input'  # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model'  # os.environ['SM_MODEL_DIR']
output_dir = '../output'  # os.environ['SM_OUTPUT_DATA_DIR']

def NN_model(model, args,train=None, target=None):
  seed_everything(args.seed)
  criterion=nn.BCEWithLogitsLoss()
  device='cuda' if torch.cuda.is_available() else 'cpu'
  model=model.to(device)
  skf=StratifiedKFold(n_splits=10,shuffle=True, random_state=args.seed)
  for fold, (train_idx, valid_idx) in enumerate(skf.split(train, target)):
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5, patience=8, threshold=1e-4)
    x_tr, x_val =train.iloc[train_idx,3:].values, train.iloc[valid_idx,3:].values
    target_tr,target_val=target[train_idx].values, target[valid_idx].values

    train_data= train_dataset(x_tr, target_tr)
    valid_data=train_dataset(x_val, target_val)
    train_dl=DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    vaild_dl=DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    best_auc=0
    for epoch in range(args.epochs):
      print(f'Epoch {epoch}/{args.epochs}')
      print('*'*20)
      for phase in ['train','valid']:
        if phase =='train':
          model.train()
        else:
          model.eval()
        running_loss, auc_score, num_cnt=0,0,0
        auc_list,label_list=list(),list() 
        if phase=='train':
          for sample in train_dl:
            inputs, label=sample['x'],sample['y']
            label=label.reshape[label.shape[0],1]
            inputs=inputs.to(device)
            label=label.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
              outputs=model(inputs)
              loss=criterion(ouputs,label)
              loss.backward()
              optimizer.step()
              running_loss+=loss.item() 
              auc_list.extend(outputs.detach().cpu())
              label_list.extend(label.detach().cpu())

        if pahse=='valid':
          for sample in valid_dl:
            inputs, labels=sample['x'],sample['y']
            labels=labels.reshape(labels.shape[0],1)
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
              outputs=model(inputs)
              loss=criterion(outputs,labels)
              running_loss+=loss.item()
              auc_list.extend(outputs.detach().cpu())
              label_lsit.extend(labels.detach().cpu())
        auc_score=roc_auc_score(label_list,auc_list)
        running_loss=float(running_loss)
        if phase=='valid':
          scheduler.step(running_loss)
        print(f'{phase}_Loss: {running_loss}, auc_score={auc_score}')

        if phase=='valid' and auc_score >= best_acc:
          best_auc=auc_score
          best_model=copy.deepcopy(model.state_dict())
          torch.save(best_model, f'./best_model{fold}.pth')
          print('best_model saved')


          

def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10):
    features = combinations(features, len(features))
    best_oof = 0
    for feature in tqdm(features):
        feature = feature
        x_train = train[feature]
        x_test = test[feature]

        # 테스트 데이터 예측값을 저장할 변수
        test_preds = np.zeros(x_test.shape[0])

        # Out Of Fold Validation 예측 데이터를 저장할 변수
        y_oof = np.zeros(x_train.shape[0])

        # 폴드별 평균 Validation 스코어를 저장할 변수
        score = 0

        # 피처 중요도를 저장할 데이터 프레임 선언
        fi = pd.DataFrame()
        fi['feature'] = feature

        # Stratified K Fold 선언
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
            # train index, validation index로 train 데이터를 나눔
            x_tr, x_val = x_train.loc[tr_idx,
                                      feature], x_train.loc[val_idx, feature]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

            # LightGBM 데이터셋 선언
            dtrain = lgb.Dataset(x_tr, label=y_tr)
            dvalid = lgb.Dataset(x_val, label=y_val)

            # LightGBM 모델 훈련
            clf = lgb.train(
                model_params,
                dtrain,
                valid_sets=[dtrain, dvalid],  # Validation 성능을 측정할 수 있도록 설정
                categorical_feature=categorical_features,
                verbose_eval=200
            )

            # Validation 데이터 예측
            val_preds = clf.predict(x_val)

            # Validation index에 예측값 저장
            y_oof[val_idx] = val_preds

            # 폴드별 Validation 스코어 측정
            # print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
            # print('-'*80)

            # score 변수에 폴드별 평균 Validation 스코어 저장
            score += roc_auc_score(y_val, val_preds) / folds

            # 테스트 데이터 예측하고 평균해서 저장
            test_preds += clf.predict(x_test) / folds

            # 폴드별 피처 중요도 저장
            fi[f'fold_{fold+1}'] = clf.feature_importance()

            gc.collect()

        print(f"\nMean AUC = {score}")  # 폴드별 Validation 스코어 출력
       # Out Of Fold Validation 스코어 출력
        print(f"OOF AUC = {roc_auc_score(y, y_oof)}")

        # 폴드별 피처 중요도 평균값 계산해서 저장
        fi_cols = [col for col in fi.columns if 'fold_' in col]
        fi['importance'] = fi[fi_cols].mean(axis=1)
        if best_oof < roc_auc_score(y, y_oof):
            best_oof = roc_auc_score(y, y_oof)
            with open('log1.txt', 'w') as f:
                f.write(str(best_oof))
                f.write(str(feature))
            print('best_roc_auc_score Saved')
        return y_oof, test_preds, fi


if __name__ == '__main__':

    # 인자 파서 선언
    parser = argparse.ArgumentParser()

    # baseline 모델 이름 인자로 받아서 model 변수에 저장
    parser.add_argument('--model', type=str, default='light_gbm',
                        help="set model light_gbm , nn_model")
    parser.add_argument('--num_features',type=int, default=55)
    praser.add_argument('--hidden_size' ,type=int, default=1024)
    parser.add_argument('--batch_size',type=int, default=512)
    parser.add_argument('--seed',type=int, default=777)
    
    args = parser.parse_args()
    model = args.model
    print('baseline model:', model)

    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])
    print(data.info())

    # 예측할 연월 설정
    year_month = '2011-12'
    train, test, y, features = feature_engineering1(data, year_month)
    if model == 'light_gbm':  # baseline 모델 3
      model_params = {
          'objective': 'binary',  # 이진 분류
          'boosting_type': 'gbdt',
          'metric': 'auc',  # 평가 지표 설정
          'feature_fraction': 0.8,  # 피처 샘플링 비율
          'bagging_fraction': 0.8,  # 데이터 샘플링 비율
          'bagging_freq': 1,
          'n_estimators': 10000,  # 트리 개수
          'early_stopping_rounds': 100,
          'seed': SEED,
          'verbose': -1,
          'n_jobs': -1,
      }
      # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
      y_oof, test_preds, fi = make_lgb_oof_prediction(
          train, y, test, features, model_params=model_params)
    elif model=='nn_model':
      model=Model(num_features=args.feature, hidden_size=args.hidden_size,num_targets=1)
      NN_model(model, args,train=train,target=y)
    # 테스트 결과 제출 파일 읽기
    sub = pd.read_csv(data_dir + '/sample_submission.csv')
    # 테스트 예측 결과 저장
    sub['probability'] = test_preds
    os.makedirs(output_dir, exist_ok=True)
    # 제출 파일 쓰기
    sub.to_csv(os.path.join(output_dir, 'output2.csv'), index=False)
