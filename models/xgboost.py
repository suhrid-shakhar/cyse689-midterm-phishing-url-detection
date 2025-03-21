from xgboost import XGBClassifier

def xgboost_model(X_train, y_train, X_test, y_test):
    model = XGBClassifier(
        eval_metric='logloss',
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        max_depth=4,  # Reduced depth to prevent overfitting
        n_estimators=100,  # Fewer trees
        reg_alpha=0.5,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        early_stopping_rounds=10,  # Early stopping
        use_label_encoder=False
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    return model, y_pred