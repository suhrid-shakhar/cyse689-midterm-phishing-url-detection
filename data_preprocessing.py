from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data(test_size=0.2):
    local_csv_path = "PhiUSIIL_Phishing_URL_Dataset.csv"

    try:
        # Try loading the dataset from a local file and remove the FILENAME column as it is not included while fetching from repository
        df = pd.read_csv(local_csv_path)
        if "FILENAME" in df.columns:
            df = df.drop(columns=["FILENAME"])

    except FileNotFoundError:
        # If the file does not exist, fetch from UCI repository and save it

        print("Local CSV not found. Fetching dataset from UCI repository...")
        phishing_data = fetch_ucirepo(id=967)

        # Convert to pandas DataFrame
        df = pd.concat([phishing_data.data.features, phishing_data.data.targets], axis=1)
        
        # Save to CSV for future use
        df.to_csv(local_csv_path, index=False)

    # Drop irrelevant columns
    drop_columns = ['URL', 'Domain', 'Title']
    df = df.drop(columns=drop_columns)

    # Encode categorical column 'TLD'
    le = LabelEncoder()
    df['TLD'] = le.fit_transform(df['TLD'])

    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']

    # Scale numerical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    print("Training datasize: ", len(X_train))
    print("Testing datasize: ", len(X_test))

    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()

    # Print class distribution
    print("Training set class distribution:")
    print(f"Class 0: {train_counts[0]}, Class 1: {train_counts[1]}")

    print("\nTesting set class distribution:")
    print(f"Class 0: {test_counts[0]}, Class 1: {test_counts[1]}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()
  

def plot_learning_curve(estimator, X, y, title, cv=5, n_jobs=-1):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 20), scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Testing Accuracy")
    plt.title(f"Learning Curve - {title}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()#block=False)

# # metadata 
# print(phiusiil_phishing_url_website.metadata) 
  
# # variable information 
# print(phiusiil_phishing_url_website.variables) 

# # Print dataset size
# print(f"Total samples: {len(X)}")

# # Check class distribution
# print("Class distribution:")
# print(y.value_counts())  # Count occurrences of each class