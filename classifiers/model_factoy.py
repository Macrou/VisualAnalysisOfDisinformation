from classifiers.model_handler import ModelHandler
from classifiers.k_neighbors import *
from classifiers.random_forest import *
from classifiers.linear_regression import *
from classifiers.xgb_model import *

class ModelFactory():
    """Factory that creates the model handler.
    """
    def __init__(self, train_features, train_labels, test_features, test_labels,models,device = 'cpu'):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.models = models
        self.device = device
 
    def create(self):
        """Creates a handler

        Returns:
            _type_: The handler_
        """
        model_handler = ModelHandler({})
        model_handler.add_model('Logistic',LinearRegressionModel(self.train_features,self.train_labels,self.test_features,self.test_labels,self.models['Logistic']))
        model_handler.add_model('Random Forest',RandomForestModel(self.train_features,self.train_labels,self.test_features,self.test_labels,self.models['Random Forest']))
        model_handler.add_model('KNN',KNeighborsModel(self.train_features,self.train_labels,self.test_features,self.test_labels,self.models['KNN']))
        model_handler.add_model('XGBoost',XgbModel(self.train_features,self.train_labels,self.test_features,self.test_labels,self.models['XGBoost'],self.device))
        return model_handler