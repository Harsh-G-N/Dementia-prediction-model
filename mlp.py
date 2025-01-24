import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

# Set seed for reproducibility
np.random.seed(42)

# Load dataset
file_path = 'oasis2.xlsx'  # Replace with your dataset path
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

# Define MLP model using scikit-learn
def create_mlp_model():
    """
    Create an MLP model with Scikit-learn's MLPClassifier.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(30, 60, 60, 20),
        learning_rate_init=0.001,
        max_iter=500,  # Increased maximum iterations
        random_state=42,
        solver='adam',
        activation='relu',
        early_stopping=False  # Disable early stopping
    )
    return model

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mlp_accuracies = []
mlp_confusion_matrices = []
mlp_classification_reports = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train the MLP model
    mlp_model = create_mlp_model()
    mlp_model.fit(X_train_resampled, y_train_resampled)

    # Evaluate the MLP model
    y_pred = mlp_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlp_accuracies.append(accuracy)
    mlp_confusion_matrices.append(confusion)
    mlp_classification_reports.append(report)

# Print results
print(f"Mean Accuracy: {np.mean(mlp_accuracies):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(mlp_accuracies):.4f}")
print("Confusion Matrices:")
for idx, matrix in enumerate(mlp_confusion_matrices, 1):
    print(f" Fold {idx}:")
    print(matrix)
    print("-" * 30)

# Print mean class-wise accuracy
classwise_accuracies = {}
for report in mlp_classification_reports:
    for class_label, metrics in report.items():
        if class_label.isdigit():
            if class_label not in classwise_accuracies:
                classwise_accuracies[class_label] = []
            classwise_accuracies[class_label].append(metrics['precision'])

print("Mean Class-wise Accuracy:")
for class_label, accuracies in classwise_accuracies.items():
    print(f" Class {class_label}: Accuracy = {np.mean(accuracies):.4f}")
