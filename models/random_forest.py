
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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
    random_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid,
                                   n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(train_features, train_labels)
    return rf_random.best_estimator_

def test(model,train_features ,train_labels,test_features, test_labels):
    model.fit(train_features, train_labels)
    # predict the mode
    test_pred = model.predict(test_features)
    # performance evaluatio metrics
    print(classification_report(test_pred, test_labels))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(model, test_features, test_labels, ax=ax, alpha=0.8)
    rfc_disp.plot(ax=ax, alpha=0.8)
    plt.show()


def train_and_test_random_forest(train_features,train_labels,test_features,test_labels):
    model = train(RandomForestClassifier(),train_features,train_labels)
    test(model,train_features,train_labels,test_features,test_labels)
