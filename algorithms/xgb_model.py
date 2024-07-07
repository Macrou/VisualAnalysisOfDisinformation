""" Random forest model trained and tested. And plotted.  

Returns:
    _type_: _description_
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import torch
from xgboost import XGBRFClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from algorithms.simple_model import SimpleModel

class XgbModel(SimpleModel):
    def __init__(self, train_features, train_labels, test_features, test_labels, model=None,device = 'cuda'):
        super().__init__(train_features, train_labels, test_features, test_labels, model)
        self.device = device
    def train(self):
        """Function for training the model for a Random Forest Classifier.

        Args:
            model (sklearn.ensemble.RandomForestClassifier): the classifier used for this training
            train_features (numpy.ndarray): _description_
            train_labels (numpy.ndarray): _description_

        Returns:
            sklearn.ensemble.RandomForestClassifier: Classifier with parameters tuned.  
        """
        print('==> Training Random forest')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        param_dist = {
            'n_estimators':[int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
            'learning_rate': np.linspace(0.01, 0.2, 10),
            'subsample': np.linspace(0.6, 1.0, 5),
            'colsample_bynode': np.linspace(0.5, 1.0, 5),
            'min_child_weight': np.arange(1, 6, 1)
        }
        model = XGBRFClassifier(random_state=42,device=self.device)       
        grid_search = RandomizedSearchCV(estimator = model, param_distributions = param_dist, 
                                        n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)

        grid_search.fit(cp.array(self.train_features), self.train_labels)
        self.model = grid_search.best_estimator_

    def test(self):
        """Tests a model with the given test models.

        Args:
            model (sklearn.ensemble.RandomForestClassifier): _description_
            train_features (numpy.ndarray): _description_
            train_labels (numpy.ndarray): _description_
            test_features (numpy.ndarray): _description_
            test_labels (numpy.ndarray): _description_
        """
        self.model.fit(self.train_features, self.train_labels)
        # predict the mode
        self.predictions = self.model.predict(self.test_features)
    

    def train_and_test(self):
        """Trains and tests a random forest algorithm

        Args:
            train_features (_type_): _description_
            train_labels (_type_): _description_
            test_features (_type_): _description_
            test_labels (_type_): _description_
        """
        start_time = time.time()
        print(f'starts time is {start_time}')
        model = self.train()
        print(model)
        total_time = time.time() - start_time
        print(f'end time is {total_time}')
        self.test(model)

    def evaluate_results(self):
        print(self.model)
        print(classification_report(self.predictions, self.test_labels))
        ax = plt.gca()
        rfc_disp = RocCurveDisplay.from_estimator(self.model, self.test_features, self.test_labels, ax=ax, alpha=0.8)
        rfc_disp.plot(ax=ax, alpha=0.8)
        plt.savefig(fname='results/plots/RocCurveRandomForest.png',format='png')
        plt.clf()
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(self.model,self.test_features,self.test_labels,normalize='true')
        plt.title('Normalized Confusion Matrix')
        plt.savefig(fname='results/plots/ConfusionMatrix.png',format='png')
        plt.clf()
        importance = self.model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.savefig(fname='results/plots/FeatureImportance.png',format='png')