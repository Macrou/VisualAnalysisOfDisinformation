
def train():
    kf=KFold(n_splits=5,shuffle=True,random_state=42)
    parameter={'n_neighbors': np.arange(2, 30, 1)}
    knn=KNeighborsClassifier()
    knn_cv=GridSearchCV(knn, param_grid=parameter, cv=kf, verbose=1)
    knn_cv.fit(X_train, y_train)