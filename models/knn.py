from sklearn.neighbors import KNeighborsClassifier

def knn_model(X_train, y_train, X_test):
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred