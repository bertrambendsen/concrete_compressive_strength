import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_excel("data/Concrete_Data.xls")  

# Loading data, and dviding it in classes
X = df.iloc[:, :-1].values
y_contious = df.iloc[:, -1].values 
median_strength = np.median(y_contious)
y = (y_contious > median_strength).astype(int)

# Regulazation parameters
lambd = np.linspace(0.01, 2, 6)  
neuron = [2, 4, 8, 16, 32]

maxiter = 100
random = 42 # To make the same results as last

# Standardizing data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# kfold
kf = KFold(n_splits=10, shuffle=True, random_state=random)

# Saveing results
results = {
    "Fold": [],
    "LR_Param": [], "LR_Error": [],
    "ANN_Param": [], "ANN_Error": [],
    "Baseline_Error": []
}

# Outer loop
fold_idx = 1
for train_idx, test_idx in kf.split(X):

    # Splits datav for outer k-fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    n_test = len(y_test)

    # Logistic Regression - Inner loop for lambda selection
    lr_best_error = float('inf')
    lr_best_lambd = None

    for lam in lambd:
        inner_errors = []

        for inner_train_idx, inner_val_idx in kf.split(X_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
            lr = LogisticRegression(C=1/lam, max_iter=maxiter)
            lr.fit(X_inner_train, y_inner_train)
            y_pred = lr.predict(X_inner_val)
            error = 1 - accuracy_score(y_inner_val, y_pred)
            inner_errors.append(error)
        avg_error = np.mean(inner_errors)

        if avg_error < lr_best_error:
            lr_best_error = avg_error
            lr_best_lambd = lam
    
    # Evaluate best logistic regression on test set
    lr = LogisticRegression(C=1/lr_best_lambd, max_iter=maxiter)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_error = np.sum(y_pred_lr != y_test) / n_test

    # ANN - Inner loop for neuron selection
    ann_best_error = float('inf')
    ann_best_neurons = None

    for neurons in neuron:
        inner_errors = []

        for inner_train_idx, inner_val_idx in kf.split(X_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
            ann = MLPClassifier(hidden_layer_sizes=(neurons,4), max_iter=maxiter, random_state=random)
            ann.fit(X_inner_train, y_inner_train)
            y_pred = ann.predict(X_inner_val)
            error = 1 - accuracy_score(y_inner_val, y_pred)
            inner_errors.append(error)
        avg_error = np.mean(inner_errors)

        if avg_error < ann_best_error:
            ann_best_error = avg_error
            ann_best_neurons = neurons
    
    # Check best ANN on testset
    ann = MLPClassifier(hidden_layer_sizes=(ann_best_neurons,4), max_iter=maxiter, random_state=random)
    ann.fit(X_train, y_train)
    y_pred_ann = ann.predict(X_test)
    ann_error = np.sum(y_pred_ann != y_test) / n_test

    # Baseline model, guessing the most occuring class
    majority_class = np.bincount(y_train).argmax()
    y_pred_baseline = np.full(n_test, majority_class)
    baseline_error = np.sum(y_pred_baseline != y_test) / n_test

    # Store results
    results["Fold"].append(fold_idx)
    results["LR_Param"].append(lr_best_lambd)
    results["LR_Error"].append(lr_error)
    results["ANN_Param"].append(ann_best_neurons)
    results["ANN_Error"].append(ann_error)
    results["Baseline_Error"].append(baseline_error)
    fold_idx += 1

# Makes data fram for results
results_df = pd.DataFrame(results)
results_df.loc["Average"] = ["Average", "", results_df["LR_Error"].mean(), 
                            "", results_df["ANN_Error"].mean(), results_df["Baseline_Error"].mean()]
print(results_df.to_string(index=False))

print(f"Logistic Regression avg error: {results_df['LR_Error'].mean():.3f}")
print(f"ANN avg error: {results_df['ANN_Error'].mean():.3f}")
print(f"Baseline avg error: {results_df['Baseline_Error'].mean():.3f}")