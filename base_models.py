import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

# Set seed for reproducibility
np.random.seed(42)

# Load dataset
file_path = r'C:\Users\Harsh\OneDrive\Desktop\python\S6-MiniProject\oasis2.xlsx'  # Replace with your dataset path
data = pd.read_excel(file_path)

# Data preprocessing
data_cleaned = data.drop(columns=['Subject ID', 'MRI ID', 'Hand'])
label_encoder = LabelEncoder()
data_cleaned['M/F'] = label_encoder.fit_transform(data_cleaned['M/F'])
data_cleaned['Group'] = label_encoder.fit_transform(data_cleaned['Group'])
data_cleaned = data_cleaned.dropna()
scaler = MinMaxScaler()
numerical_features = [col for col in data_cleaned.columns if col not in ['M/F', 'Group']]
data_cleaned[numerical_features] = scaler.fit_transform(data_cleaned[numerical_features])

X = data_cleaned.drop(columns=['Group'])
y = data_cleaned['Group']

# Define base models
base_models = {
    "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in base_models.items():
    accuracies = []
    confusion_matrices = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Train the model
        model.fit(X_train_resampled, y_train_resampled)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        accuracies.append(accuracy)
        confusion_matrices.append(confusion)

    # Store results for the model
    results[name] = {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "confusion_matrices": confusion_matrices
    }

# Print the results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Mean Accuracy: {result['mean_accuracy']:.4f}")
    print(f"Standard Deviation of Accuracy: {result['std_accuracy']:.4f}")
    print("Confusion Matrices:")
    for idx, matrix in enumerate(result['confusion_matrices'], 1):
        print(f" Fold {idx}:")
        print(matrix)
        print("-" * 30)
    print("=" * 50)
