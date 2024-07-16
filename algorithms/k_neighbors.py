import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from algorithms.simple_model import SimpleModel
import pickle

class KNeighborsModel(SimpleModel):
    def train(self):
        print('==> Training KNN')
        parameter={'n_neighbors': np.arange(2, 30, 1)}
        knn=KNeighborsClassifier()
        knn_cv=RandomizedSearchCV(estimator = knn, param_distributions = parameter, 
                                        n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = 8)
        knn_cv.fit(self.train_features, self.train_labels)
        self.model = knn_cv.best_estimator_
        
    def test(self):
        self.model.fit(self.train_features, self.train_labels)
        filename = './results/checkpoint/finalized_k_model.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        self.predictions = self.model.predict(self.test_features)

    def train_and_test(self):
        self.train()
        self.test()
        
    def evaluate_results(self):
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
   
    def save_correct_incorrect_predictions(self):
        correct_indices = np.where(self.predictions == self.test_labels)[0]
        incorrect_indices = np.where(self.predictions != self.test_labels)[0]
 