from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from utility.cmdlineManagement.datasetSelection import DatasetSelection

# Select dataset
dataframe = DatasetSelection().dataset

# Define the parameter grid
param_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create a base model
rf = RandomForestClassifier()

# Instantiate the random search model
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42)#, n_jobs=-1)

# Definisci le features (X) e il target (Y) cioè la variabile da prevedere

X = dataframe.drop(['user'], axis=1)
y = dataframe["user"]

# Fit the random search model
rf_random.fit(X, y)

# Print the best parameters
print("Best Parameters: ", rf_random.best_params_)

# Print the best score
print("Best Score: ", rf_random.best_score_)

# You can also get the results for all the tried combinations
results = rf_random.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(params, "Mean CV Score: ", mean_score)