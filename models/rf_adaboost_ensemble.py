from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC  # You can also use any other classifier you like

def rf_adaboost_ensemble(X_train, y_train, X_test):
    # Define individual classifiers
    rf_model = RandomForestClassifier(n_estimators=50)
    ada_model = AdaBoostClassifier(n_estimators=50)
    svm_model = SVC(probability=True)  # Adding an SVM model for voting

    # Create the voting ensemble model (hard voting by default)
    ensemble_model = VotingClassifier(estimators=[
        ('rf', rf_model),
        ('ada', ada_model),
        ('svm', svm_model)
    ], voting='hard')  # 'hard' means majority voting, use 'soft' for probability voting

    # Fit the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Make predictions
    y_pred = ensemble_model.predict(X_test)
    
    return ensemble_model, y_pred
