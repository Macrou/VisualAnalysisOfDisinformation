from algorithms.model_handler import ModelHandler
from algorithms.k_neighbors import *
from algorithms.random_forest import *
from algorithms.linear_regression import *
from algorithms.xgb_model import *

class ModelFactory():
    def __init__(self, train_features, train_labels, test_features, test_labels,model=None,device = 'cpu'):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.model = model
        self.device = device
 
    def create(self):
        model_handler = ModelHandler({})
        model_handler.add_model('Logistic',LinearRegressionModel(self.train_features,self.train_labels,self.test_features,self.test_labels,self.model))
        model_handler.add_model('Random Forest',RandomForestModel(self.train_features,self.train_labels,self.test_features,self.test_labels,self.model))
        model_handler.add_model('KNN',KNeighborsModel(self.train_features,self.train_labels,self.test_features,self.test_labels,self.model))
        model_handler.add_model('XGBoost',XgbModel(self.train_features,self.train_labels,self.test_features,self.test_labels,self.model,self.device))
        return model_handler