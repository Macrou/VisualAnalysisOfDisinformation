
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from algorithms.simple_model import SimpleModel

class LinearRegressionModel(SimpleModel):
    
    def train(self):
        print('==> Training logistic regression')
        grid = {
            'solver':['newton-cg', 'lbfgs', 'liblinear'],
            'penalty': ['l2'],
            'C': [100, 10, 1.0, 0.1, 0.01],
            'max_iter' : [400]
        }
        knn= LogisticRegression()
        model_cv=GridSearchCV(knn, param_grid=grid, cv=5, verbose=1)
        model_cv.fit(self.train_features, self.train_labels)
        self.model = model_cv.best_estimator_
        
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
        self.predictions = self.model.predict(self.test_features)

    def train_and_test(self):
        """Trains and tests a linear regression model.
        Args:
            train_features (_type_): _description_
            train_labels (_type_): _description_
            test_features (_type_): _description_
            test_labels (_type_): _description_
        """
        self.train()
        print(self.model)
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
