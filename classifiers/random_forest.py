""" Random forest model trained and tested. And plotted.  

Returns:
    _type_: _description_
"""
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from classifiers.simple_model import SimpleModel
import pickle

class RandomForestModel(SimpleModel):
    """Random forest model. 

    Args:
        SimpleModel (SimpleModel): a simple model
    """
    def train(self):
        print('==> Training Random forest')
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
        max_depth = [int(x) for x in np.linspace(10, 50, num = 11)]
        param_grid = {
            'bootstrap': [True],
            'max_depth': max_depth,
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [1, 2, 4],
            'n_estimators': n_estimators
        }
        model = RandomForestClassifier(n_jobs = -1)
        grid_search = RandomizedSearchCV(estimator = model, param_distributions = param_grid, 
                                        n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

        grid_search.fit(self.train_features, self.train_labels)
        self.model = grid_search.best_estimator_

    def test(self):
        self.model.fit(self.train_features, self.train_labels)
        filename = './results/checkpoint/finalized_random_forest_model.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        
        # predict the mode
        self.predictions = self.model.predict(self.test_features)
        

    def train_and_test(self):
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