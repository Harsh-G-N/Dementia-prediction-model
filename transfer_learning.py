import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier  # Scikit-learn MLP Classifier
import pandas as pd

# Set seeds for reproducibility
np.random.seed(42)

# Load dataset
file_path = 'oasis2.xlsx'
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
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

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

# Step 1: Data splitting and SMOTE
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
class_accuracy = {str(cls): [] for cls in np.unique(y)}  # Store accuracy for each class (as strings)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Step 2: Train base models and collect predictions on training data
    base_predictions_train = []
    base_predictions_test = []
    for name, model in base_models.items():
        model.fit(X_train_resampled, y_train_resampled)
        base_predictions_train.append(model.predict_proba(X_train_resampled))
        base_predictions_test.append(model.predict_proba(X_test))

    # Combine base model predictions
    X_meta_train = np.hstack(base_predictions_train)
    X_meta_test = np.hstack(base_predictions_test)

    # Step 3: Train MLP model on meta features
    mlp_model = create_mlp_model()
    mlp_model.fit(X_meta_train, y_train_resampled)

    # Step 4: Evaluate final MLP model on test data
    y_pred = mlp_model.predict(X_meta_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    results.append({"accuracy": accuracy, "confusion_matrix": confusion})

    # Calculate per-class accuracy
    report = classification_report(y_test, y_pred, output_dict=True)
    for cls, metrics in report.items():
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:  # Skip non-class entries
            class_accuracy[str(cls)].append(metrics['f1-score'])  # Store f1-score or accuracy per class as string

# Aggregate results
mean_accuracy = np.mean([res['accuracy'] for res in results])
std_accuracy = np.std([res['accuracy'] for res in results])
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")

# Display the mean accuracy for each class
for cls, acc_list in class_accuracy.items():
    mean_class_acc = np.mean(acc_list)
    print(f"Mean accuracy for class {cls}: {mean_class_acc:.4f}")

print("Confusion Matrices:")
for idx, res in enumerate(results, 1):
    print(f" Fold {idx}:")
    print(res['confusion_matrix'])
    print("-" * 30)
