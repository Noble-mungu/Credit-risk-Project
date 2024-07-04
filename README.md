
## Customer Risk Model

Customer Risk Model is designed to predict customer risk using various machine learning algorithms. The model uses cross-validation to avoid overfitting and evaluates its performance using the F1 score due to the imbalance in the dataset.

## Dependencies
The following Python libraries are required to run the model:

statistics
sklearn (including Pipeline, cross_val_score, StratifiedKFold, and cross_val_predict)
matplotlib
numpy
You can install these dependencies using pip:

`
pip install numpy matplotlib scikit-learn
`
## Model Building
The model building process involves creating three different classifiers:

Logistic Regression
K-Nearest Neighbors (KNN)
Random Forest
These classifiers are evaluated using cross-validation to ensure robustness and avoid overfitting.

`

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# Define models
lr = LogisticRegression()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()

# List of models and their names
models = [lr, knn, rf]
model_names = ['Logistic Regression', 'KNN', 'Random Forest']

# Lists to store results
results, mean_results, predictions, f1_test_scores = [], [], [], []

`

## Model Fitting, Cross-Validation, and Performance Evaluation
The `algor` function performs the following steps for each model:

* Creates a pipeline with the model.
* Fits the model to the training data.
* Uses StratifiedKFold for cross-validation.
* Evaluates the model using the F1 score.
* Stores and prints the results.

`
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
import statistics as st
import numpy as np

def algor(model, model_name):
    print('\n', model_name)
    # Create pipeline
    pipe = Pipeline([('model', model)])
    pipe.fit(x_train, y_train)
    
    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=5)
    
    # Cross-validation scores
    n_scores = cross_val_score(pipe, x_train, y_train, scoring='f1_weighted', cv=cv, n_jobs=-1, error_score='raise')
    results.append(n_scores)
    mean_results.append(st.mean(n_scores))
    
    print('F1-Score (train): mean=%.3f, min=%.3f, max=%.3f, stdev=%.3f' % (st.mean(n_scores), min(n_scores), max(n_scores), np.std(n_scores)))
    
    # Predictions and F1 score on test data
    y_pred = cross_val_predict(model, x_train, y_train, cv=cv)
    predictions.append(y_pred)
    f1 = f1_score(y_train, y_pred, average='weighted')
    f1_test_scores.append(f1)
    
    print('F1-Score (test): %.4f' % f1)


# Iterate over each model and evaluate

for model, model_name in zip(models, model_names):
    algor(model, model_name)

`

## Model Comparison and Visualization
Finally, the performance of the models is compared and visualized using a box plot of the F1 scores from cross-validation.

`
import matplotlib.pyplot as plt

# Visualize model performance

fig = plt.figure(figsize=(18, 15))
plt.title('Model Evaluation by Cross-Validation Method')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.boxplot(results, labels=model_names, showmeans=True)
plt.show()
`
