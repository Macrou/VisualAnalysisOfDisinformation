import torch
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets 

import clip
from dataloaders.fakeddit_data_loader import Fakeddit


from tqdm import tqdm
import os
import argparse

from algorithms.model_factoy import ModelFactory

from dataloaders.fakeddit_data_loader import *
parser = argparse.ArgumentParser(description='PyTorch Disinformation Training')
parser.add_argument('--data','-d', choices=['Fakeddit', 'CIFAKE'],default='Fakeddit', type= str, help='data set')
parser.add_argument('--classifier','-c',choices=['Logistic', 'Random Forest','KNN','XGBoost'],default='Logistic',type= str, help='classifier used')
#parser.add_argument('--mode','-m',choices=['Logistic', 'Random Forest','KNN',],default='Logistic',type= str, help='mode used')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print('==> Building model..')
model, preprocess = clip.load('ViT-B/32', device) 
if args.data == 'Fakeddit':  
    train = Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_train.tsv",transform=preprocess)
    test =  Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_test_public.tsv",transform=preprocess)
elif args.data == 'CIFAKE':
    train = datasets.ImageFolder(root='dataset/train',transform=preprocess)
    test = datasets.ImageFolder(root='dataset/test',transform=preprocess)



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
    model_handler = ModelFactory(train_features,train_labels,test_features,test_labels,device).create()
    model_handler.train_model(args.classifier)