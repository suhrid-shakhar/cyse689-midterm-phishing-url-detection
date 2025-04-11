from xgboost import XGBClassifier

def xgboost_model(X_train, y_train, X_test):
    model = XGBClassifier(
        eval_metric='logloss',
        max_depth=4, 
        n_estimators=100, 
        reg_alpha=0.1, 
        reg_lambda=1.0 
    )
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    return model, y_pred