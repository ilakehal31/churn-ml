import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
import time
import optuna
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

OUTPUT_PATH = Path("./out/score.txt")

# Concat and load the dataset
df = pd.concat(
    [
        pd.read_csv('data/customer_churn_dataset-training-master.csv'), 
        pd.read_csv('data/customer_churn_dataset-testing-master.csv')
    ], 
    axis=0)
df.reset_index(drop=True, inplace=True)

# Dropping the CustomerID column and removing rows with missing values
df.drop(columns='CustomerID', inplace=True)
df[df.isna().any(axis=1)]
df.dropna(inplace=True) 

df.shape
print(df.head())

y = df['Churn']
X = df.drop(columns='Churn')

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_features),
    (make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'), OneHotEncoder(handle_unknown='ignore')), categorical_features)
)

# First baseline with the model LogisticRegression
def first_baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(
        preprocessor,
        LogisticRegression()
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Result of the baseline with Logistic Regression:")
    print(classification_report(y_test, y_pred))
    

first_baseline(X, y)

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted')
}

results = []

# Iterate through each models, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, HistGradientBoostingClassifier
def create_models():
    return [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        XGBClassifier(eval_metric='logloss'),
        HistGradientBoostingClassifier()
    ]

# Function to perform k-fold cross-validation and calculate metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
def k_fold_cross_validation_with_metrics(model_class, X, y, preprocessor, k_folds=5):
    stratified_kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    train_scores = []
    test_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc_roc': []
    }

    for train_index, test_index in stratified_kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline = make_pipeline(preprocessor, model_class)
        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        test_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        train_scores.append(accuracy_score(y_train, train_pred))
        test_scores['accuracy'].append(accuracy_score(y_test, test_pred))
        test_scores['precision'].append(precision_score(y_test, test_pred, average='weighted'))
        test_scores['recall'].append(recall_score(y_test, test_pred, average='weighted'))
        test_scores['f1'].append(f1_score(y_test, test_pred, average='weighted'))
        test_scores['auc_roc'].append(roc_auc_score(y_test, test_pred_proba))

    mean_train_score = np.mean(train_scores)
    mean_test_accuracy = np.mean(test_scores['accuracy'])
    mean_test_precision = np.mean(test_scores['precision'])
    mean_test_recall = np.mean(test_scores['recall'])
    mean_test_f1 = np.mean(test_scores['f1'])
    mean_test_auc_roc = np.mean(test_scores['auc_roc'])

    return {
        'Train Accuracy': mean_train_score,
        'Test Accuracy': mean_test_accuracy,
        'Test Precision': mean_test_precision,
        'Test Recall': mean_test_recall,
        'Test F1': mean_test_f1,
        'Test AUC-ROC': mean_test_auc_roc,
        'Overfitting Gap': mean_train_score - mean_test_accuracy
    }


# Perform k-fold cross-validation and calculate metrics for each models
# for model_class in create_models():
#     model_name = model_class.__class__.__name__
    
#     start_time = time.time()
    
#     metrics = k_fold_cross_validation_with_metrics(model_class, X, y, preprocessor)
        
#     results.append({
#         'Model': model_name,
#         **metrics
#     })

# results_df = pd.DataFrame(results)
# results_df = results_df.sort_values('Test Accuracy', ascending=False)

# print(results_df)

# Find the best hyper-parameters to train the model HistGradientBoostingClassifier
# def objective(trial):
#     params = {
#         'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
#         'max_iter': trial.suggest_int('max_iter', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
#         'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 100),
#         'l2_regularization': trial.suggest_float('l2_regularization', 1e-10, 1.0, log=True)
#     }
    
#     model = make_pipeline(
#         preprocessor,
#         HistGradientBoostingClassifier(**params, random_state=42)
#     )
    
#     return cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

# Optimize hyper-parameters with Optuna
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# Best hyper-parameters find:  {'learning_rate': 0.14086510295122037, 'max_iter': 948, 'max_depth': 10, 'min_samples_leaf': 30, 'max_leaf_nodes': 89, 'l2_regularization': 7.478168083575335e-09}
# Best score:  0.9360002395646816

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Churn Prediction")

# Function to evaluate the model HistGradientBoostingClassifier
def evaluate_model(model, X, y, preprocessor):
    metrics = k_fold_cross_validation_with_metrics(model, X, y, preprocessor)
    return metrics

# Function to write the results in a score file
def write_results_to_file(base_metrics, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Model Results\n")
        f.write("=============\n\n")
        f.write("HistGradientBoostingClassifier Results:\n")
        for metric, value in base_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

# Register the run
with mlflow.start_run(run_name="Base HistGradientBoostingClassifier"):
    base_model = HistGradientBoostingClassifier(random_state=42)
    base_metrics = evaluate_model(base_model, X, y, preprocessor)
    
    mlflow.log_params(base_model.get_params())
    for metric_name, metric_value in base_metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    mlflow.sklearn.log_model(base_model, "base_model")

    print("\nResults of the base model HistGradientBoostingClassifier:")
    for metric, value in base_metrics.items():
        print(f"  {metric}: {value:.4f}")

# Second best model: RandomForestClassifier
with mlflow.start_run(run_name="Second Best - RandomForestClassifier"):
    rf_model = RandomForestClassifier(random_state=42)
    rf_metrics = evaluate_model(rf_model, X, y, preprocessor)
    
    mlflow.log_params(rf_model.get_params())
    for metric_name, metric_value in rf_metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    print("\nRésultats du RandomForestClassifier:")
    for metric, value in rf_metrics.items():
        print(f"  {metric}: {value:.4f}")

# Comparison between the two models
print("\nComparaison (RandomForest - Base):")
for metric in base_metrics.keys():
    diff = rf_metrics[metric] - base_metrics[metric]
    print(f"  {metric}: {diff:.4f}")


# Write results to file
write_results_to_file(base_metrics, OUTPUT_PATH)

print(f"Results have been written to {OUTPUT_PATH}")
