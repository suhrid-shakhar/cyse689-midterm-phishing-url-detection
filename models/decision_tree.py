from sklearn.tree import DecisionTreeClassifier

def decision_tree_model(X_train, y_train, X_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred