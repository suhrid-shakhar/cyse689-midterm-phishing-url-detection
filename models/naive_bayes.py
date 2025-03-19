from sklearn.naive_bayes import GaussianNB

def naive_bayes_model(X_train, y_train, X_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred