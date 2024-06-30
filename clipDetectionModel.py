import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets 
import torchvision.transforms as transforms

import clip
from dataloaders.fakeddit_data_loader import Fakeddit

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import argparse

from models import random_forest

from dataloaders.fakeddit_data_loader import *
parser = argparse.ArgumentParser(description='PyTorch Disinformation Training')
parser.add_argument('--data','-d', default='Fakeddit', type= str, help='data set')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print('==> Building model..')
model, preprocess = clip.load('ViT-B/32', device) 
train =  Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_train.tsv",transform=preprocess)
test = Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_test_public.tsv",transform=preprocess)
# train = datasets.ImageFolder(root='dataset/train',transform=preprocess)
# test = datasets.ImageFolder(root='dataset/test',transform=preprocess)



classes = ('false','real')


def get_features(dataset):
    all_features = []
    all_labels = []
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            images = images.to(device)
            features = model.encode_image(images)
            all_features.append(features)
            all_labels.append(labels)      
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


if __name__ == "__main__":
    print('==> Getting features..')
    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)
    print('==> Training random forest model..')
    random_forest.train_and_test_random_forest(train_features,train_labels,test_features,test_labels)