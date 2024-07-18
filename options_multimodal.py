import argparse

parser = argparse.ArgumentParser(description='PyTorch Disinformation Training')
parser.add_argument('--classifier','-c',choices=['Logistic', 'Random Forest','KNN','XGBoost'],default='Logistic',type= str, help='classifier used')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
