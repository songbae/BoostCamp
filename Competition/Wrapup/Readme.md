### Source code 
-train.py 
-3ways to train 3models 
in kernel you can train by some args added

```
~$ python train.py --kfold_use 0 --resize 224 --seed 777 --average macro --train_transform train_tfms_mask --class_num 3 --epochs 10 --batch_size 64 --model_num 1 --optimizer Adam --loss crossentropy_loss --model_name mask --age_num 58
```

you can you `help` 

kfold_use : witch kfold you are going to use 
resize : image input size 
seed : default 777 , for reproduction 
average: you can use one of macro or binary (in gender there is only two class so must use binary)
train_transform : For Mask-> train_tfms_mas  For Age,gender -> train_age_gender 
class_num : how many classes for output -> Mask :3  Age : 3 Gender :2 
epoch : default 10 
batch_size : default 64
model_num: mask :1  age:2  gender:3
optimzer: default Adam 
loss : default crossentropy_loss either you can use focal_loss 
model_name: dafault mask ->it save the best_model_pth by the model_name
age_num : filter for age default 58 


Test.py 
---
```
~$ python test.py --name sumission_ --tta tes
```
name: will be your sumission.csv name 
tta : default=yes it determines use tta method or not . if not using tta just input 'no'

log
---
there are many models.pth and submission codes that i tried  submission which indicates 'best' is the best-f1_score 

model_folder
---
it's not useful code don't mind never using it 


utils_folder
---
Never used folder dont'y mind 

data_folder
---
Train&Valid : PstageDataset-> used for train.py 
Customaugmentation: used for aumgentation method 
PstageDataset : used for test.py (not using tta ) 
tta_dataset: used for test.py(using tta)
