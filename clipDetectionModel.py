from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from sklearn.linear_model import LogisticRegression
import clip
from xgboost import XGBRFClassifier
from dataloaders.fakeddit_data_loader import Fakeddit
import pickle

from tqdm import tqdm
import argparse

from algorithms.model_factoy import ModelFactory

from dataloaders.fakeddit_data_loader import *
parser = argparse.ArgumentParser(description='PyTorch Disinformation Training')
parser.add_argument('--data','-d', choices=['Fakeddit', 'CIFAKE'],default='Fakeddit', type= str, help='data set')
parser.add_argument('--classifier','-c',choices=['Logistic', 'Random Forest','KNN','XGBoost'],default='Logistic',type= str, help='classifier used')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

filenames = {
        'Logistic':  './checkpoint/finalized_logistic_regression_model.sav',
        'Random Forest': './checkpoint/finalized_random_forest_model.sav',
        'KNN': './checkpoint/finalized_k_model.sav',
        'XGBoost': './checkpoint/finalized_random_forest_model.sav'
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print('==> Building model..')
model, preprocess = clip.load('ViT-B/32', device) 
if args.data == 'Fakeddit':  
    train = Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_train.tsv",transform=preprocess)
    test =  Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_test_public.tsv",transform=preprocess)
    plaintest = Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_test_public.tsv")
elif args.data == 'CIFAKE':
    train = datasets.ImageFolder(root='dataset/train',transform=preprocess)
    test = datasets.ImageFolder(root='dataset/test',transform=preprocess)
    plaintest = datasets.ImageFolder(root='dataset/test',transform=transforms.Resize((512,512)))


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
    #train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)
    models = {
        'Logistic': LogisticRegression(C=10, max_iter=400, solver='newton-cg',n_jobs=-1),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=29,n_jobs=-1),
        'XGBoost': XGBRFClassifier()
    }
    if args.resume:
        models[args.classifier] = pickle.load(open(filenames[args.classifier],'rb'))
    model_handler = ModelFactory(None,None,test_features,test_labels,models=models,device=device).create()
    #model_handler.train_model(args.classifier)
    model_handler.save_correct_incorrect_predictions_model(args.classifier,plaintest)
    #model_handler.evaluate_results(args.classifier)