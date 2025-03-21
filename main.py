from data_preprocessing import load_and_preprocess_data, plot_learning_curve
from models import logistic_regression, svm, xgboost, random_forest, naive_bayes, knn, decision_tree, mutual_info_logreg, rf_adaboost_ensemble
from evaluation import evaluate_model

X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()  

print("\nRunning Logistic Regression")
model, y_pred = logistic_regression.logistic_regression_model(X_train, y_train, X_test)
plot_learning_curve(model, X_train, y_train, title='Logistic Regression', cv=5)
evaluate_model(y_test, y_pred, "Logistic Regression")

print("\nRunning SVM")
model, y_pred = svm.svm_model(X_train, y_train, X_test)
evaluate_model(y_test, y_pred, "SVM")

print("\nRunning XGBoost")
model, y_pred = xgboost.xgboost_model(X_train, y_train, X_test, y_test)
evaluate_model(y_test, y_pred, "XGBoost")

print("\nRunning Random Forest")
model, y_pred = random_forest.random_forest_model(X_train, y_train, X_test)
evaluate_model(y_test, y_pred, "Random Forest")

print("\nRunning Naive Bayes")
model, y_pred = naive_bayes.naive_bayes_model(X_train, y_train, X_test)
evaluate_model(y_test, y_pred, "Naive Bayes")

print("\nRunning KNN")
model, y_pred = knn.knn_model(X_train, y_train, X_test)
evaluate_model(y_test, y_pred, "KNN")

print("\nRunning Decision Tree")
model, y_pred = decision_tree.decision_tree_model(X_train, y_train, X_test)
evaluate_model(y_test, y_pred, "Decision Tree")

print("\nRunning Mutual Info + Logistic Regression")
model, y_pred, selected = mutual_info_logreg.mutual_info_logistic_regression(X_train, y_train, X_test)
print(f"Selected Feature Indices: {selected}")
evaluate_model(y_test, y_pred, "Mutual Info + Logistic Regression")

print("\nRunning Random Forest + AdaBoost Ensemble")
model, y_pred = rf_adaboost_ensemble.rf_adaboost_ensemble(X_train, y_train, X_test)
evaluate_model(y_test, y_pred, "RF + AdaBoost Ensemble")
