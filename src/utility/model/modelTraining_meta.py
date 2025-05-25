import os
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn_genetic.callbacks import ProgressBar
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utility.model.model_utilities import ModelUtilities as mu

class MetaLearner:
    def __init__(self, dataset, label_column='user',
                 output_name=None, model_path="../models/", retrain=False):
        self.dataset = mu.check_not_none(dataset, "dataset")
        self.label_column = mu.check_not_none(label_column, "label_column")
        self.model_path = mu.check_path(model_path)

        self.output_name = mu.check_output_model_name("meta_", ".joblib", output_name)
        mu.check_duplicate_model_name(self.output_name, retrain, self.model_path)

        self.retrain = retrain if retrain else False

        self.param_grid = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(3, 10),
            'learning_rate': Continuous(0.01, 0.3),
            'subsample': Continuous(0.6, 1.0),
            'colsample_bytree': Continuous(0.6, 1.0),
            'min_child_weight': Integer(1, 10),
            'gamma': Continuous(0, 0.5),
            'reg_alpha': Continuous(0, 10),
            'reg_lambda': Continuous(0.1, 10)
        }

        self.model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='mlogloss',
            random_state=42
        )
        

    def prepare_data(self, test_size=0.2, val_size=0.25, random_state=42):
        """Prepara i dati per il meta-training."""

        X = self.dataset.drop(columns=[self.label_column]).values
        y = self.dataset[self.label_column].values

        X_test = None
        y_test = None
        
        # Split data into train, validation, and test sets
        if test_size > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_temp = X
            y_temp = y
            
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")

        if X_test:
            print(f"Test set: {X_test.shape[0]} samples")

        # Check class distribution
        train_class_counts = np.bincount(y_train)
        val_class_counts = np.bincount(y_val)

        print("Class distribution in training set:", train_class_counts)
        print("Class distribution in validation set:", val_class_counts)

        if y_test:
            test_class_counts = np.bincount(y_test)
            print("Class distribution in test set:", test_class_counts)
        

        return X_train, y_train, X_val, y_val, X_test, y_test

    # def train(self, X_train, y_train, X_val, y_val):
    #     """Train the meta-learner."""

    #     best_acc = 0.0
    #     best_model = None
    #     best_params = None

    #     param_count = 1
    #     grid  = ParameterGrid(self.param_grid)
    #     param_number = len(grid)

    #     for params in tqdm(grid, desc=f"Finding best parameters {param_count}/{param_number}"):
    #         model = LogisticRegression(**params, random_state=42)
    #         model.fit(X_train, y_train)
    #         acc = model.score(X_val, y_val)
    #         print(f"Params: {params} -> Validation Acc: {acc:.4f}")
    #         if acc > best_acc:
    #             best_acc = acc
    #             best_model = model
    #             best_params = params

    #         param_count += 1

    #     self.model = best_model
    #     print(f"Best Params: {best_params}\nBest Validation Accuracy: {best_acc:.4f}")
    #     return best_params

    def train(self, X_train, y_train, X_val, y_val):
        
        evolved_estimator = GASearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=5,
            scoring='accuracy',
            population_size=20,
            generations=10,
            tournament_size=3,
            elitism=True,
            crossover_probability=0.8,
            mutation_probability=0.1,
            verbose=True,
            n_jobs=-1
        )

        progress = ProgressBar()

        evolved_estimator.fit(X_train, y_train, callbacks=progress)
        
        self.model = evolved_estimator.best_estimator_

        acc = evolved_estimator.best_score_
        return acc
        
    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)

        accuracy = metrics.accuracy_score(y_test, preds)
        classification_report = metrics.classification_report(y_test, preds, output_dict=True)
        confusion_matrix = metrics.confusion_matrix(y_test, preds)

        # Handle binary vs. multi-class AUC
        n_classes = probs.shape[1]
        if n_classes == 2:
            # take probability of the “positive” class
            roc_auc = metrics.roc_auc_score(y_test, probs[:, 1])
        else:
            # multi-class OVR
            roc_auc = metrics.roc_auc_score(y_test, probs, multi_class='ovr')

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

    def prepare_and_train(self, val_size=0.25, random_state=42):
        print("Preparing data...")
        X_train, y_train, X_val, y_val, _, _ = self.prepare_data(test_size=0)

        print("Training meta-learner...")
        return self.train(X_train, y_train, X_val, y_val)


    def train_and_evaluate(self, test_size=0.2, val_size=0.25, random_state=42, plot_results=True):
        """Train and evaluate the meta-learner."""

        print("Preparing data...")
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(test_size=test_size, val_size=val_size, random_state=random_state)

        print("Training meta-learner...")
        best_params = self.train(X_train, y_train, X_val, y_val)

        print("Evaluating meta-learner...")
        accuracy, classification_report, roc_auc, confusion_matrix = self.evaluate(X_test, y_test)

        if plot_results:
            print("Plotting metrics...")
            self.plot_metrics(classification_report, confusion_matrix)

        return {
            'accuracy': accuracy,
            'classification_report': classification_report,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix,
            'best_params': best_params
        }
        
    
    def save_model(self):
        save_path = os.path.join(self.model_path, self.output_name)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")
        
           
        

         
             


        
        


