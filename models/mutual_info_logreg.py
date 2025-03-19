from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression

def mutual_info_logistic_regression(X_train, y_train, X_test):
    selector = SelectKBest(mutual_info_classif, k=10)  # Select top 10 features
    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_new, y_train)
    y_pred = model.predict(X_test_new)
    return model, y_pred, selector.get_support(indices=True)
