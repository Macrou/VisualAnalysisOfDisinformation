import clip.simple_tokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets 
import torchvision.transforms as transforms
from numpy.linalg import norm


import clip
from multimodal_fakeddit_data_loader import Multimodal_Fakeddit

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os
import argparse

from models import *

#from fakedditDataLoader import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device) 

#train = datasets.ImageFolder(root='dataset/cifake/train',transform=preprocess)
#test = datasets.ImageFolder(root='dataset/cifake/test',transform=preprocess)

train =  Multimodal_Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_train.tsv",transform=preprocess)
test = Multimodal_Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_validate.tsv",transform=preprocess)

classes = ('false','real')


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images,text, labels in tqdm(DataLoader(dataset, batch_size=100)):
            image_features = model.encode_image(images.to(device))
            text_features = model.encode_text(text)
            features =
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
print('==> Getting features ')
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)
scores = {}

print('==> Classifying images')


for C in (10**k for k in range(-6, 6)):
    classifier = LogisticRegression(random_state=0, C=C, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    scores[C] = {'train accuracy': classifier.score(train_features, train_labels), 
                 'test accuracy': classifier.score(test_features, test_labels)}

fig, axs = plt.subplots(figsize=(12, 4))    
pd.DataFrame.from_dict(scores, 'index').plot(ax=axs,logx=True, xlabel='C', ylabel='accuracy');
plt.savefig(fname='results/plots/trainingAccuracy.png',format='png')