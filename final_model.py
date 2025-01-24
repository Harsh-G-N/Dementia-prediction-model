import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  # Scikit-learn MLP Classifier
from mealpy import FloatVar, GWO
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

def create_mlp_model(params):
    """
    Create an MLP model with Scikit-learn's MLPClassifier.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(int(params[0]), int(params[1]), int(params[2])),
        learning_rate_init=params[3],
        max_iter=500,  # Increased maximum iterations
        random_state=42,
        solver='adam',
        activation='relu',
        early_stopping=True,  # Stop early if no improvement
        n_iter_no_change=10,  # Number of epochs with no improvement
        validation_fraction=0.2  # 20% of training data used for validation
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

    # Further split resampled data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.2, random_state=42, stratify=y_train_resampled
    )

    # Step 2: Train base models and collect predictions on training and validation sets
    base_predictions_train = []
    base_predictions_val = []
    for name, model in base_models.items():
        model.fit(X_train_split, y_train_split)
        base_predictions_train.append(model.predict_proba(X_train_split))
        base_predictions_val.append(model.predict_proba(X_val))

    # Combine base model predictions
    X_meta_train = np.hstack(base_predictions_train)
    X_meta_val = np.hstack(base_predictions_val)

    # Step 3: Hyperparameter optimization for MLP using validation data
    def objective_function(solution):
        n_neurons_1 = int(solution[0])
        n_neurons_2 = int(solution[1])
        n_neurons_3 = int(solution[2])
        learning_rate = solution[3]

        params = [n_neurons_1, n_neurons_2, n_neurons_3, learning_rate]
        mlp_model = create_mlp_model(params)

        mlp_model.fit(X_meta_train, y_train_split)
        accuracy = mlp_model.score(X_meta_val, y_val)
        return -accuracy  # Minimize negative accuracy

    problem_dict = {
        "bounds": FloatVar(
            lb=(10, 10, 10, 0.0001),
            ub=(200, 200, 200, 0.01),
            name="mlp_params"
        ),
        "minmax": "min",
        "obj_func": objective_function
    }

    model_gwo = GWO.GWO_WOA(epoch=10, pop_size=20)
    g_best = model_gwo.solve(problem_dict)

    # Step 4: Train optimized MLP model on full training data (meta features)
    best_params = g_best.solution
    mlp_model = create_mlp_model(best_params)
    mlp_model.fit(X_meta_train, y_train_split)

    # Step 5: Evaluate final MLP model on test data
    base_predictions_test = [model.predict_proba(X_test) for model in base_models.values()]
    X_meta_test = np.hstack(base_predictions_test)

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
