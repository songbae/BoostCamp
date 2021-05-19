import os
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import copy
import pandas as pd
import datetime as dt

# 시드 고정 함수


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# 평가 지표 출력 함수


def print_score(label, pred, prob_thres=0.5):
    print('Precision: {:.5f}'.format(
        precision_score(label, pred > prob_thres)))
    print('Recall: {:.5f}'.format(recall_score(label, pred > prob_thres)))
    print('F1 Score: {:.5f}'.format(f1_score(label, pred > prob_thres)))
    print('ROC AUC Score: {:.5f}'.format(roc_auc_score(label, pred)))


def make_product_month(df):
    df = df.copy()
    df['year_month'] = pd.to_datetime(df['order_date']).dt.strftime('%Y-%m')
    df['product_category'] = data['product_id'].str[:3]


def make_month_over_train_3OO(df):
    data = df.copy()
    data['year_month'] = pd.to_datetime(
        data['order_date']).dt.strftime('%Y-%m')
    label = sorted(data['customer_id'].unique())
    temp = data.groupby(['customer_id', 'year_month'])['total'].sum()
    temp = pd.DataFrame(temp)
    temp = temp.reset_index()
    print(temp.head())
    list3, list6, list9, list12, list20 = list(), list(), list(), list(), list()
    start_month3, start_month6, start_month9, start_month12, start_month20, end_month = '2011-06', '2011-03', '2010-12', '2010-09', '2010-01', '2011-11'
    for i in label:
        month3, month6, month9, month12, month20 = 0, 0, 0, 0, 0
        for j in temp.values:
            if j[0] == i and (start_month3 < j[1] < end_month):
                if j[2] >= 300:
                    month3 += 1/4
            if j[0] == i and (start_month6 < j[1] < end_month):
                if j[2] >= 300:
                    month6 += 1/7
            if j[0] == i and (start_month9 < j[1] < end_month):
                if j[2] >= 300:
                    month9 += 1 / 10
            if j[0] == i and (start_month12 < j[1] < end_month):
                if j[2] >= 300:
                    month12 += 1 / 13
            if j[0] == i and (start_month20 < j[1] < end_month):
                if j[2] >= 300:
                    month20 += 1 / 21
        list3.append(month3),list6.append(month6),list9.append(month9),list12.append(month12),list20.append(month20)
    temp1 = pd.DataFrame(list3, columns=['month_4'])
    temp2 = pd.DataFrame(list6, columns=['month_7'])
    temp3 = pd.DataFrame(list9, columns=['month_10'])
    temp4 = pd.DataFrame(list12, columns=['month_13'])
    temp5 = pd.DataFrame(list20, columns=['month_20'])
    list_df = pd.concat([temp1, temp2, temp3, temp4, temp5], axis=1)
    list_df.to_csv('./input/month_over_300_train1.csv', index=False)


def make_month_over_test_3OO(df):
    data = df.copy()
    data['year_month'] = pd.to_datetime(
        data['order_date']).dt.strftime('%Y-%m')
    label = sorted(data['customer_id'].unique())
    temp = data.groupby(['customer_id', 'year_month'])['total'].sum()
    temp = pd.DataFrame(temp)
    temp = temp.reset_index()
    print(temp.head())
    list3, list6, list9, list12, list20 = list(), list(), list(), list(), list()
    start_month3, start_month6, start_month9, start_month12, start_month20, end_month = '2011-07', '2011-04', '2011-01', '2010-10', '2010-02', '2011-12'
    for i in label:
        month3, month6, month9, month12, month20 = 0, 0, 0, 0, 0
        for j in temp.values:
            if j[0] == i and (start_month3 < j[1] < end_month):
                if j[2] >= 300:
                    month3 += 1/4
            if j[0] == i and (start_month6 < j[1] < end_month):
                if j[2] >= 300:
                    month6 += 1/7
            if j[0] == i and (start_month9 < j[1] < end_month):
                if j[2] >= 300:
                    month9 += 1 / 10
            if j[0] == i and (start_month12 < j[1] < end_month):
                if j[2] >= 300:
                    month12 += 1 / 13
            if j[0] == i and (start_month20 < j[1] < end_month):
                if j[2] >= 300:
                    month20 += 1 / 21
        list3.append(month3),list6.append(month6),list9.append(month9),list12.append(month12),list20.append(month20)
    temp1 = pd.DataFrame(list3, columns=['month_4'])
    temp2 = pd.DataFrame(list6, columns=['month_7'])
    temp3 = pd.DataFrame(list9, columns=['month_10'])


def make_time_corr_train(df):
  data = pd.read_csv('./input/train.csv', parse_dates=['order_date'])
  data['order_time'] = data.order_date.dt.strftime('%H')
  temp = data[data['order_date'] < '2011-11']
  temp['order_time'] = temp['order_time'].astype(int)
  temp = temp.groupby('customer_id')[['order_time']].mean()
  temp.reset_index(inplace=True)
  temp['order_time'] = np.round(temp['order_time'])
  temp.to_csv('./input/order_time_train.csv', index=False)



def make_time_corr_test(df):
  data = pd.read_csv('./input/train.csv', parse_dates=['order_date'])
  data['order_time'] = data.order_date.dt.strftime('%H')
  temp = data[data['order_date'] < '2011-12']
  temp['order_time'] = temp['order_time'].astype(int)
  temp = temp.groupby('customer_id')[['order_time']].mean()
  temp.reset_index(inplace=True)
  temp['order_time'] = np.round(temp['order_time'])
  temp.to_csv('./input/order_time_test.csv', index=False)


