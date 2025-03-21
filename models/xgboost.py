from xgboost import XGBClassifier

def xgboost_model(X_train, y_train, X_test):
    model = XGBClassifier(
        eval_metric='logloss',
        max_depth=4,  # Reduced depth to prevent overfitting
        n_estimators=100,  # Fewer trees
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0  # L2 regularization
    )
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    return model, y_pred