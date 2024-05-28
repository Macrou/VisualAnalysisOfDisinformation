import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.utils
import torch.utils
from torchvision import datasets 
import torchvision.transforms as transforms

import clip

import numpy as np

import os
import argparse

from models import *
from utils import progress_bar, get_mean_and_std
import matplotlib.pyplot as plt
from fakedditDataLoader import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device) 

train = datasets.ImageFolder(root='dataset/cifake/train',transform=preprocess)
test = datasets.ImageFolder(root='dataset/cifake/test',transform=preprocess)


classes = ('false','real')


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

