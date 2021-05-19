### Tabular Data code 
---
### inference.py
```
~$ python infernence.py --model --num_features --hidden_size --batch_size --seed 
```
### args:
  - model: [light_bgm, nn_model ,type=str, default=light_bgm] 
  > select which model to use 

  - num_features: [type=int , default=55] 
  > select feature nums 

  - hidden_size: [type=int, default=1024] 
  > how many nodes to use in hidden layer 

  - batch_size:[type=int, defalut=512] 
  > select batch_size

  - seed:[type=int, default=777] 
  > seed num for reproduction 

--- 

### utils.py 
  - seed_everything
  > set seed for reproduction

  - print_score
  > matric for evaluating score 
  - make_product_month
  > make obj-> numeric tpye of product_id with group_by month
  - make_month_over_train_300
  > feature extraction by bins of 3,6,9,20 month with groupby [customer_id ,year_month] ratio of over 300$
  - make_month_over_test_300
  > same with above but using year_month 2011-11
  - make_time_corr_train
  > feature extraction by bins of order_data 

  - make-time _corr_train
  > same with above
---
### features.py 

- generate_label
```
return label
```
> making labels by 2011-11 
- make_feature 
```
return x_tr, x_te
```
> transform obj-> numeric tpye and fill None values with Median values 
- feature_engineering1
```
return : x_tr, x_te, all_train_data['label'],features
```
> adding features and agg function for features 



