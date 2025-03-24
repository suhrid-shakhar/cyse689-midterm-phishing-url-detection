from sklearn.linear_model import LogisticRegression

def logistic_regression_model(X_train, y_train, X_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred