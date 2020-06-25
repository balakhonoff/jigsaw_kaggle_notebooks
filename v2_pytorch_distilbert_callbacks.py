#!/usr/bin/env python
# coding: utf-8

# **Pytorch BERT baseline**

# In this version, I convert https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic into pytorch version

# **Please upvote the kernel if you find it helpful**

# ### Install HuggingFace transformers & sacremoses dependency

# As we are not allowed to use internet I've created required datasets and commands to setup Hugging Face Transformers setup in offline mode. You can find the required github codebases in the datasets.
# 
# * sacremoses dependency - https://www.kaggle.com/axel81/sacremoses
# * transformers - https://www.kaggle.com/axel81/transformers

# In[1]:


# !pip install ./sacremoses/sacremoses-master/
# !pip install ./transformers/transformers-master/
STRIDE = 1
FASTPART= False


# ### Required Imports
# 
# I've added imports that will be used in training too

# In[2]:


from sklearn.utils import shuffle
from datetime import datetime
import pandas as pd
pd.set_option('display.max.columns', 500)
import numpy as np
import os
# os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.1/lib64'
import matplotlib.pyplot as plt
import gc
from shutil import copyfile
from catalyst.dl import SupervisedRunner, AlchemyLogger, CriterionCallback
from catalyst.dl.callbacks.metrics import AUCCallback
from torch.utils.data import DataLoader, SubsetRandomSampler,Dataset
batch_size = 14
token = "d1dd16f08d518293bcbeddd313b49aa4"
DATA_DIR = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'


# In[3]:


if os.uname()[1] == 'kb-Z370P-D3':
    # desktop
    LOG_PATH = '/media/ssd/logs/jigsaw'
    SERVER = False
    print('Working on desktop')
elif os.uname()[1] == 'kb-server':
    # server
    LOG_PATH = '/home/kb/logs/jigsaw'
    SERVER = True
    print('Working on server')
else:
    raise Exception('which hostname???')
    


# In[4]:


df_train_toxic = pd.read_csv(DATA_DIR+'jigsaw-toxic-comment-train.csv')
df_train_toxic.head()


# In[5]:


df_train_bias = pd.read_csv(DATA_DIR + 'jigsaw-unintended-bias-train.csv')
df_train_bias.head()


# In[6]:


# len(df_train_toxic[df_train_toxic['toxic']==1]),len(df_train_toxic[df_train_toxic['toxic']==0])
# len(df_train_bias[df_train_bias['toxic']<0.1])/7,len(df_train_bias[df_train_bias['toxic']>0.80])


# In[7]:


df_train_bias_pos = df_train_bias[df_train_bias['toxic']>0.8].reset_index(drop=True)
df_train_bias_neg = shuffle(df_train_bias[df_train_bias['toxic']<0.1].reset_index(drop=True)).reset_index(drop=True).iloc[::7]
df_train_bias =  df_train_bias_pos.append(df_train_bias_neg).reset_index(drop=True)


# In[8]:


df_valid = pd.read_csv(DATA_DIR + 'validation.csv')
df_valid.head()


# In[9]:


df_valid.groupby(['toxic', 'lang']).count()


# In[10]:


df_test = pd.read_csv(DATA_DIR + 'test.csv')
df_test.head()


# In[11]:


df_test.groupby(['lang']).count()


# In[12]:


len(df_test)


# In[13]:


df_sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')
df_sub.head()


# In[14]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[15]:


target_columns = 'toxic'


# ### Define dataset

# In[16]:


from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import time
from torch.optim import lr_scheduler

import torch
from tqdm import tqdm
#import torch.utils.data as data
from torchvision import datasets, models, transforms
from transformers import *
import random
from math import floor, ceil
from sklearn.model_selection import GroupKFold

MAX_LEN = 512
SEP_TOKEN_ID = 102

class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, df, train_mode=True, labeled=True):
        self.df = df
        if train_mode:
            self.labels = df.toxic.values
            
        self.train_mode = train_mode
        self.labeled = labeled
        #self.tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')
#         self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        #distilbert-base-multilingual-cased

    def __getitem__(self, index):
        row = self.df.iloc[index]
        token_ids = self.get_token_ids(row)
        
        if self.labeled:
            labels = self.get_label(row)
            return {'features': token_ids, 'targets': labels}

        else:
            return {'features': token_ids}

    def __len__(self):
        return len(self.df)

    def trim_input(self, text, max_sequence_length=MAX_LEN):
        t = self.tokenizer.tokenize(text)
        t_len = len(t)

        if t_len + 2 > max_sequence_length:

            t_new_len = int(max_sequence_length) - 2

            t = t[:t_new_len]

        return t
        
    def get_token_ids(self, row):
        t_tokens = self.trim_input(row.comment_text)

#         tokens = ['[CLS]'] + t_tokens  + ['[SEP]']+ t_tokens[-1::-1]+ ['[SEP]']
        tokens = ['[CLS]'] + t_tokens  + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        if len(token_ids) < MAX_LEN:
            token_ids += [0] * (MAX_LEN - len(token_ids))
            
        ids = torch.tensor(token_ids)
        
        return ids

    def get_label(self, row):
#         label = torch.tensor(row[target_columns].astype(np.long))
        label = np.round(row[target_columns])
        return torch.tensor([1-label, label]).float()
    
    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])

        if self.labeled:
            labels = torch.stack([x[1] for x in batch])
            return {'features': token_ids, 'targets': labels}
        else:
            return {'features': token_ids}


# ## Build Model

# In[17]:


from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestModel(nn.Module):
    def __init__(self, n_classes=2):
        super(QuestModel, self).__init__()
        self.model_name = 'QuestModel'
        
#         self.bert_model = BertModel.from_pretrained('bert-base-uncased') 
#         self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
    
        self.fc = nn.Linear(768, n_classes)

    def forward(self, ids):
        attention_mask = (ids > 0)
        layers = self.bert_model(input_ids=ids, attention_mask=attention_mask)
        
        out = F.dropout(layers[-1][:, 0, :], p=0.2, training=self.training)
        logit = self.fc(out)#.unsqueeze(1)
        return logit #, 'for_auc': logit[:, 1]}#[:,1]
    


# In[18]:


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))             if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)             if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            dataset.labels[idx]
#             raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


# In[19]:


dataframes = {
    'bias': df_train_bias,
    'toxic': df_train_toxic,
    'valid': df_valid,
    'test': df_test
}
def get_loaders(train_pt='bias', valid_pt='valid', test_pt='toxic', to_balance=True, shuffle_before=True):
    workers = 6
    
    if isinstance(train_pt, list):
        df_train = dataframes[train_pt[0]][['comment_text', 'toxic']]
        for pt in train_pt[1:]:
            df_train = df_train.append(dataframes[pt][['comment_text', 'toxic']]).reset_index(drop=True)
    else:
        df_train = dataframes[train_pt][['comment_text', 'toxic']]
    
    if shuffle_before:
        df_train = shuffle(df_train)
    
    train_dataset = QuestDataset(df_train.iloc[::STRIDE], train_mode=True)
    valid_dataset = QuestDataset(dataframes[valid_pt], train_mode=False)
    test_dataset = QuestDataset(dataframes[test_pt].iloc[::STRIDE], train_mode=False)
    
    train_loader = DataLoader(
        train_dataset,
        num_workers=workers,
        sampler=ImbalancedDatasetSampler(train_dataset) if to_balance else None,
        batch_size=batch_size,
    )
    valid_loader = DataLoader(
        valid_dataset,
        num_workers=workers,
        batch_size=batch_size,
    )    
    
    test_loader = DataLoader(
        test_dataset,
        num_workers=workers,
        batch_size=batch_size,
    )
       
    loaders = {}
    loaders['train'] = train_loader
    
    loaders['valid'] = valid_loader
    
    loaders['test'] = test_loader
    
    for i in ['es', 'it', 'tr']:
        df = dataframes[valid_pt]
        df = df[df['lang']==i]
        loaders['valid_'+ i] = DataLoader(
            QuestDataset(df, train_mode=False),
            num_workers=workers,
            batch_size=batch_size,
        )
    
    
    return loaders


# In[20]:


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:



project = "jigsaw_v1_distilbert"
num_epochs = 5

group =f'v2_1_new_metrics_auc'+ datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

if FASTPART:
    group = f'fast_{group}'

variants = {
    'learn_toxic':{
        'train': 'toxic',
        'valid': 'valid',
        'test': 'bias',
    },
    'learn_bias':{
        'train': 'bias',
        'valid': 'valid',
        'test': 'toxic',
    },
    'learn_both':{
        'train': ['bias', 'toxic'],
        'valid': 'valid',
        'test': 'valid'
    },
    
}
gradient_accumulation_steps = 1

lr =3e-5# 0.0001
group = group.replace('.', '')

for experiment in variants.keys():
    
    logdir = f"{LOG_PATH}/{project}/{group}/{experiment}"

    model = QuestModel(2)

    model = model.to(device)

    # model runner
    runner = SupervisedRunner(input_key=('features'), input_target_key=('targets'), output_key=('logits'))


    loaders = get_loaders(variants[experiment]['train'], 
                          variants[experiment]['valid'],
                          variants[experiment]['test'], to_balance=True)
            
    
    t_total = len(loaders['train'])//gradient_accumulation_steps*num_epochs
    warmup_proportion = 0.01
    num_warmup_steps = t_total * warmup_proportion
    
    criterion = torch.nn.BCEWithLogitsLoss()
#     criterion = torch.nn.CrossEntropyLoss()
#     criterion = torch.nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr = lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total) 
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.25)
    print(f'----------------Experiment: {experiment}')

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        distributed=True if SERVER else False,
        callbacks=[
            AlchemyLogger(
                    token=token, # your Alchemy token
                    project=project,
                    experiment=experiment,
                    group=group,
                ),
            AUCCallback(input_key = 'targets',
                                output_key = 'logits',
                                prefix = 'auc',
                                class_names = None,
                                num_classes = 2,
                                activation = 'Sigmoid',)
        ]

    )


# In[ ]:





# In[ ]:





# In[ ]:




