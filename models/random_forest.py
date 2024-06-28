""" Random forest model trained and tested. And plotted.  

Returns:
    _type_: _description_
"""
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 3, n_jobs = 8, verbose = 2)
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
    model = train(RandomForestClassifier(),train_features,train_labels)
    print(model)
    test(model,train_features,train_labels,test_features,test_labels)
