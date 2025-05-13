from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..')))

from utility.cmdlineManagement.datasetSelection import DatasetSelection

dataset_selection = DatasetSelection()
dataset = dataset_selection.dataset
dataset_name = dataset_selection.dataset_name

label_column = "user" 

X = dataset[[col for col in dataset.columns if col != label_column]].values
y = dataset[label_column].values

# Assume `embeds` is your N×768 matrix and `y` your labels:
X_train, X_test, y_train, y_test = train_test_split(
    dataset, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression(max_iter=500).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))