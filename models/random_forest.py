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


def train(model,train_features,train_labels):
    """Function for training the model for a Random Forest Classifier.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): the classifier used for this training
        train_features (numpy.ndarray): _description_
        train_labels (numpy.ndarray): _description_

    Returns:
        sklearn.ensemble.RandomForestClassifier: Classifier with parameters tuned.  
    """
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    param_grid = {
        'bootstrap': [True],
        'max_depth': max_depth,
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [1, 2, 4],
        'n_estimators': n_estimators
    }
    grid_search = RandomizedSearchCV(estimator = model, param_distributions = param_grid, 
                                     n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

    grid_search.fit(train_features, train_labels)
    return grid_search.best_estimator_

def test(model,train_features ,train_labels,test_features, test_labels):
    """Tests a model with the given test models.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): _description_
        train_features (numpy.ndarray): _description_
        train_labels (numpy.ndarray): _description_
        test_features (numpy.ndarray): _description_
        test_labels (numpy.ndarray): _description_
    """
    model.fit(train_features, train_labels)
    # predict the mode
    test_pred = model.predict(test_features)
    # performance evaluatio metrics
    print(classification_report(test_pred, test_labels))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(model, test_features, test_labels, ax=ax, alpha=0.8)
    rfc_disp.plot(ax=ax, alpha=0.8)
    plt.savefig(fname='results/plots/RocCurveRandomForest.png',format='png')
    plt.clf()
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(model,test_features,test_labels,normalize='true')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(fname='results/plots/ConfusionMatrix.png',format='png')
    plt.clf()
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.savefig(fname='results/plots/FeatureImportance.png',format='png')

def train_and_test_random_forest(train_features,train_labels,test_features,test_labels):
    """Trains and tests a random forest algorithm

    Args:
        train_features (_type_): _description_
        train_labels (_type_): _description_
        test_features (_type_): _description_
        test_labels (_type_): _description_
    """
    start_time = time.time()
    print(f'starts time is {start_time}')
    model = train(RandomForestClassifier(n_jobs = -1),train_features,train_labels)
    print(model)
    total_time = time.time() - start_time
    print(f'end time is {total_time}')
    test(model,train_features,train_labels,test_features,test_labels)
