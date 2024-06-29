
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def train(train_features,train_labels):
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    knn= LogisticRegression
    model_cv=GridSearchCV(knn, param_grid=grid, cv=3, verbose=1)
    model_cv.fit(train_features, train_labels)
    return model_cv.best_estimator_
    
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

def train_and_test_linear_regression(train_features,train_labels,test_features,test_labels):
    """Trains and tests a linear regression model.
    Args:
        train_features (_type_): _description_
        train_labels (_type_): _description_
        test_features (_type_): _description_
        test_labels (_type_): _description_
    """
    model = train(train_features,train_labels)
    print(model)
    test(model,train_features,train_labels,test_features,test_labels)