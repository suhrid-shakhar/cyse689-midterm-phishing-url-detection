from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

def rf_adaboost_ensemble(X_train, y_train, X_test):
    # Define individual classifiers
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
   
    
    ensemble_model = VotingClassifier(estimators=[
        ('rf', rf_model),
        ('ada', ada_model)
    ], voting='hard')  # 'hard' means majority voting, use 'soft' for probability voting

    # Fit the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Make predictions
    y_pred = ensemble_model.predict(X_test)
    
    return ensemble_model, y_pred
