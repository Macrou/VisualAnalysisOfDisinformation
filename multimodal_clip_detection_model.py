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
from classifiers.model_factoy import ModelFactory
from dataloaders.multimodal_fakeddit_data_loader import Multimodal_Fakeddit

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os
import argparse
from options_clip import args

from dataloaders.multimodal_fakeddit_data_loader import Multimodal_Fakeddit

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device) 
  
train = Multimodal_Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_train.tsv",transform=preprocess,tokenizer=clip.tokenize)
test =  Multimodal_Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_test_public.tsv",transform=preprocess,tokenizer=clip.tokenize)

classes = ('false','real')


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images,texts, labels in tqdm(DataLoader(dataset, batch_size=100)):
            image_features = model.encode_image(images.to(device))
            text_features = model.encode_text(texts.to(device))
            features = torch.maximum(image_features,text_features)
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


if __name__ == "__main__":
    print('==> Getting features..')
    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)
    model_handler = ModelFactory(train_features,train_labels,test_features,test_labels,device=device).create()
    model_handler.train_model(args.classifier)
    model_handler.test_model(args.classifier)
    model_handler.evaluate_results(args.classifier)