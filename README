# Customer Churn Prediction Project

## 🤑 Business use case description

This project aims to predict customer churn for a company. 
The aim is to identify customers likely to leave the service, enabling the company to take proactive measures to retain them. 
This prediction is crucial to maintaining the customer base and optimizing retention strategies.

## 📊 Dataset description

The dataset used combines training and test data, containing information about customers such as:

- Demographic characteristics (age, gender)
- Usage data (usage frequency, support calls)
- Contractual information (subscription type, contract duration)
- Financial behavior (total expenses, payment delays)
- Customer engagement (last interaction)

You can have additional information about the dataset in the eda notebook.

The dataset contains 101,042 entries after cleaning, with a churn distribution of approximately 55% (churn) and 45% (non-churn).

## 📈 Baseline

### Used features

All available features were used, except for 'CustomerID' because it's not a relevant feature for the prediction.

### Preprocessing

- Dropping rows with missing values
- One-hot encoding for categorical variables
- Standardization of numerical variables

### Model

The baseline model used is a Logistic Regression, in this case with categorical variables.

### Metrics obtained

- Accuracy: 0.8460
- Precision: 0.87 (for the churn class)
- Recall: 0.85 (for the churn class)
- F1-score: 0.86 (for the churn class)

## 🔄 First Iteration

### Changes made

We replaced the Logistic Regression with a HistGradientBoostingClassifier.

### Reasons for the change

The HistGradientBoostingClassifier is capable of capturing complex nonlinear relationships in the data and handles categorical variables well without requiring prior encoding.

### Impact on metrics

- Accuracy: 0.9361 (9% improvement)
- Precision: 0.9423 (7% improvement)
- Recall: 0.9361 (8% improvement)
- F1-score: 0.9354 (7% improvement)
- AUC-ROC: 0.9537

The use of the HistGradientBoostingClassifier has significantly improved all metrics, indicating a better ability to predict churn with precision.

## 🚀 How to start the project

1. Clone the repository:
   ```
   git clone [URL_DU_REPO]
   ```

2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Launch the Docker containers:
   ```
   docker-compose build
   docker-compose up -d
   ```

4. Run the main script:
   ```
   python script.py
   ```

5. To visualize the results in MLflow, access:
   ```
   http://localhost:8083
   ```

This project uses MLflow for experiment tracking and Docker for environment management, ensuring reproducibility and scalability.
