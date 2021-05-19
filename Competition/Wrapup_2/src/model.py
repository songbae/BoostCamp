
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.nn.utils import weight_norm



class train_dataset:
  def __init__(self, features,targets):
    self.features=features
    self.targets=targets
  def __len__(self):
    return (self.features.shape[0])
  def __getitem__(self,idx):
    sample={
      'x': torch.tensor(self.features[idx,:], dtype=torch.float),
      'y': torch.tenssor(self.features[idx],dtype=torch.float)
    }
    return sample 

class test_dataset:
  def __init__(self, features):
    self.features=features
  def __len__(self):
    return (self.features.shape[0])
  def __getitem__(self, idx):
    sample={
      'x': torch.tensor(self.features[idx,:], dtype=torch.float)
    }
    return sample 

class Model(nn.Module):
  def __init__(self, num_features,hidden_size, num_targets):
    super(Model, self).__init__()
    self.batch_norm1 = nn.BatchNorm1d(num_features)
    self.dropout1 = nn.Dropout(0.2)
    self.dense1 = weight_norm(nn.Linear(num_features, 512))
    self.batch_norm2 = nn.BatchNorm1d(512)
    self.dropout2 = nn.Dropout(0.5)
    self.dense2 = weight_norm(nn.Linear(512, hidden_size))
    self.batch_norm3 = nn.BatchNorm1d(hidden_size)
    self.dropout3 = nn.Dropout(0.5)
    self.dense3 = weight_norm((nn.Linear(hidden_size, hidden_size)))
    #self.batch_norm4 = nn.BatchNorm1d(hidden_size)
    self.dropout4= nn.Dropout(0.2)
    self.dense4=weight_norm((nn.Linear(hidden_size,num_targets)))

  def forward(self, x):
    x = self.batch_norm1(x)
    x=self.dropout1(x)
    x = F.relu(self.dense1(x))
    
    x = self.batch_norm2(x)
    x = self.dropout2(x)
    x = F.relu(self.dense2(x))
    
    x = self.batch_norm3(x)
    x = self.dropout3(x)
    x = F.relu(self.dense3(x))
    #x = self.batch_norm4(x)
    x = self.dropout4(x)
    x=self.dense4(x)
    return x