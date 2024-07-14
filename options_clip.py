import argparse


parser = argparse.ArgumentParser(description='PyTorch Disinformation Training')
parser.add_argument('--data','-d', choices=['Fakeddit', 'CIFAKE'],default='Fakeddit', type= str, help='data set')
parser.add_argument('--classifier','-c',choices=['Logistic', 'Random Forest','KNN','XGBoost'],default='Logistic',type= str, help='classifier used')
#parser.add_argument('--mode','-m',choices=['Logistic', 'Random Forest','KNN',],default='Logistic',type= str, help='mode used')
args = parser.parse_args()
