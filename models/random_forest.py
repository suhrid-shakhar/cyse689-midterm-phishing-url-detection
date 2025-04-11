from sklearn.ensemble import RandomForestClassifier

def random_forest_model(X_train, y_train, X_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred