import os
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import ParameterGrid, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from utility.model.model_utilities import ModelUtilities as mu

class MetaLearner:
    def __init__(self, dataset, label_column='user',
                 output_name=None, model_path="../models/", retrain=False):
        self.model = None
        self.dataset = mu.check_not_none(dataset, "dataset")
        self.label_column = mu.check_not_none(label_column, "label_column")
        self.model_path = mu.check_path(model_path)

        self.output_name = mu.check_output_model_name("meta_", ".pth", output_name)
        mu.check_duplicate_model_name(self.output_name, retrain)

        self.retrain = retrain if retrain else False

        self.param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [100, 200],
            'warm_start': [True]

        }

    def prepare_data(self, test_size=0.2, val_size=0.25, random_state=42):
        """Prepara i dati per il meta-training."""

        X = self.dataset.drop(columns=[self.label_column]).values
        y = self.dataset[self.label_column].values
        
        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Check class distribution
        train_class_counts = np.bincount(y_train)
        val_class_counts = np.bincount(y_val)
        test_class_counts = np.bincount(y_test)
        
        print("Class distribution in training set:", train_class_counts)
        print("Class distribution in validation set:", val_class_counts)
        print("Class distribution in test set:", test_class_counts)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, X_train, y_train, X_val, y_val):
        """Train the meta-learner."""

        best_acc = 0.0
        best_model = None
        best_params = None
        for params in ParameterGrid(self.param_grid):
            model = LogisticRegression(**params, random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_val, y_val)
            print(f"Params: {params} -> Validation Acc: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = params

        self.model = best_model
        print(f"Best Params: {best_params}, Best Validation Accuracy: {best_acc:.4f}")
        
    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)

        accuracy = metrics.accuracy_score(y_test, preds)
        classification_report = metrics.classification_report(y_test, preds, output_dict=True)
        roc_auc = metrics.roc_auc_score(y_test, probs, multi_class='ovr')
        confusion_matrix = metrics.confusion_matrix(y_test, preds)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Classification Report:\n{classification_report}")
        print(f"Confusion Matrix:\n{confusion_matrix}")

        return accuracy, classification_report, roc_auc, confusion_matrix
    
    def plot_metrics(self, classification_report, confusion_matrix):
        """Plot metrics."""

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

        # Plot classification report
        plt.figure(figsize=(10, 7))
        sns.heatmap(pd.DataFrame(classification_report).iloc[:-1, :].T, annot=True, cmap='Blues')
        plt.title('Classification Report')
        plt.show()

    def train_and_evaluate(self, test_size=0.2, val_size=0.25, random_state=42, plot_metrics=True):
        """Train and evaluate the meta-learner."""

        print("Preparing data...")
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(test_size=test_size, val_size=val_size, random_state=random_state)

        print("Training meta-learner...")
        self.train(X_train, y_train, X_val, y_val)

        print("Evaluating meta-learner...")
        accuracy, classification_report, roc_auc, confusion_matrix = self.evaluate(X_test, y_test)

        if plot_metrics:
            print("Plotting metrics...")
            self.plot_metrics(classification_report, confusion_matrix)

        return {
            'accuracy': accuracy,
            'classification_report': classification_report,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix
        }
        
    
    def save_model(self):
        save_path = os.path.join(self.model_path, self.output_name)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")
        
           
        

         
             


        
        


