from sklearn.svm import SVC

def svm_model(X_train, y_train, X_test):
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred