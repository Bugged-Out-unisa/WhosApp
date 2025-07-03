import time
import os
import sys
import calendar
import argparse
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.special import logit
from sklearn.model_selection import train_test_split
from utility.logging import LoggerReport, LoggerUserModelHistory
from utility.model.modelTraining_feature import ModelTraining
from utility.model.modelTraining_embeddings import CNN1D, FocalLoss
from utility.model.modelTraining_meta import MetaLearner
from utility.model.model_list import models
from utility.dataset.metaDataset import MetaDataset as md
# HOW TO USE:
# py new_training.py -oN <*outputName> -c <*configFile> -st <*feature|embeddings|both|meta> -fd <*feature_dataset> -ed <*embeddings_dataset> -cv <fold_number> -r <*retrain>

# CHECKS IF SPECIFIED DATASET EXIST
# (dataCreation.py return already existing DF)

# ELSE IT CREATES A NEW DATASET WITH SPECIFIED NAME from new_dataset.py

# ONCE A DATASET IS GIVEN, IT TRAINS MODEL THEN PERSISTS IT

def check_dataset_exists(dataset_path):

    if not dataset_path.endswith(".parquet"):
        dataset_path += ".parquet"

    # Controlla se il file esiste
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Il dataset specificato non esiste: {dataset_path}")

    # Carica il dataset come DataFrame di pandas
    return pd.read_parquet(dataset_path)

if __name__ == "__main__":
    # Argomenti da linea di comando
    parser = argparse.ArgumentParser()

    timestamp = str(calendar.timegm(time.gmtime()))

    # Optional arguments
    parser.add_argument("-oN", "--outputName", help="Nome file del modello salvato" , required=False, default=timestamp)
    parser.add_argument("-c", "--config", help="File config", required=False)
    parser.add_argument("-r", "--retrain", action="store_true", help="Opzione di retraining", required=False)
    parser.add_argument("-st", "--select_training", help="selezione se eseguire training feature o embeddings", required=False, default="meta")
    parser.add_argument("-cv", "--cross_val_folds", type=int, help="Number of folds for meta-learner cross-validation", required=False, default=5)
    parser.add_argument("-fd", "--feature_dataset", help="Dataset per il training delle feature", required=False, default=None)
    parser.add_argument("-ed", "--embeddings_dataset", help="Dataset per il training delle embeddings", required=False, default=None)

    args = parser.parse_args()
    output_name, config, retrain, select_training, n_folds, feature_data_file, embeddings_data_file = args.outputName, args.config, args.retrain, args.select_training, args.cross_val_folds, args.feature_dataset, args.embeddings_dataset

    feature_training = False
    embeddings_training = False
    meta_training = False

    select_training = select_training.lower()
    if select_training not in ["feature", "embeddings", "both", "meta"]:
        raise ValueError("Invalid value for --select_training. Choose 'feature', 'embeddings', 'both' or 'meta'.")
    
    if select_training == "feature":
        feature_training = True
    elif select_training == "embeddings":
        embeddings_training = True
    elif select_training == "both":
        feature_training = True
        embeddings_training = True
    else:
        feature_training = True
        embeddings_training = True
        meta_training = True


    # LOGGING:: Inizializza il logging
    LoggerReport(
        name=output_name,
        start_message="!! NEW TRAINING !!",
        path=LoggerReport.TRAINING_LOGGING_PATH
    ).run()

    feature_dataset = None
    embeddings_dataset = None

    if feature_training:
        if feature_data_file is None:
            raise ValueError("Feature dataset file must be specified with -fd or --feature_dataset")

        print("\n-- Features --")
        
        feature_dataset = check_dataset_exists(feature_data_file)
    
    if embeddings_training:
        if embeddings_data_file is None:
            raise ValueError("Embeddings dataset file must be specified with -ed or --embeddings_dataset")

        print("\n-- Embeddings --")

        embeddings_dataset = check_dataset_exists(embeddings_data_file)

    if meta_training:
        print("\n[INFO] Getting holdout data...")

        # stratify by 'user' so label proportions stay roughly equal
        train_ids, holdout_ids = train_test_split(
            feature_dataset['message_id'],
            test_size=0.15,
            random_state=42,
            stratify=feature_dataset['user']
        )

        # Slice both DataFrames by IDs
        feature_train_dataset   = feature_dataset[feature_dataset['message_id'].isin(train_ids)].reset_index(drop=True)
        feature_holdout_dataset = feature_dataset[feature_dataset['message_id'].isin(holdout_ids)].reset_index(drop=True)

        embeddings_train_dataset   = embeddings_dataset[embeddings_dataset['message_id'].isin(train_ids)].reset_index(drop=True)
        embeddings_holdout_dataset = embeddings_dataset[embeddings_dataset['message_id'].isin(holdout_ids)].reset_index(drop=True)
    else:
        feature_train_dataset = feature_dataset
        embeddings_train_dataset = embeddings_dataset
        

    if feature_training:
        print("\n-- Feature Model --")
        # Select model
        model_choice = models["random_forest"]  # Default model
        # feature_train_dataset.drop("message_id", axis=1, inplace=True)
        # LoggerUserModelHistory.append_model_user(dataset_name, output_name)

        print(f"\n[INFO] Training model: {model_choice} with output name: {output_name}")
        # Training del modello con i parametri passati da linea di comando
        feature_model = ModelTraining(output_name, model_choice, feature_train_dataset.drop("message_id", axis=1), config, retrain)
        feature_model.train(plot_results=False)
        feature_model.save_model()

    if embeddings_training:
        # Training del modello con i parametri passati da linea di comando
        # embeddings_train_dataset.drop("message_id", axis=1, inplace=True)

        print("\n-- CNN Model --")
        print(f"\n[INFO] Training CNN model with output name: {output_name}")
        cnn = CNN1D(embeddings_train_dataset.drop("message_id", axis=1), output_name=output_name, retrain=retrain,)
        cnn.train_and_evaluate(criterion=FocalLoss(alpha=.5, gamma=4), plot_results=False)
        cnn.save_model()

    if meta_training:
        print("\n--- Meta Learner Training with Cross-Validation ---")
        print(f"\n[INFO] Using {n_folds} folds for cross-validation")

        df_meta = md.build_simple_meta_dataset(feature_train_dataset, embeddings_train_dataset, n_folds)
        df_meta_enhanced = md.enhance_meta_dataset(df_meta)

        # Train meta-learner on full cross-validated data
        meta_learner = MetaLearner(df_meta_enhanced, output_name=output_name, retrain=retrain)
        meta_learner.prepare_and_train()
        meta_learner.save_model()

        # Final evaluation on holdout set
        print("\n--- Final Evaluation on Holdout Set ---")
        
        # Get predictions from the feature model and CNN
        feature_holdout_data = feature_holdout_dataset.drop(columns=['user', 'message_id'])
        embeddings_holdout_data = embeddings_holdout_dataset.drop(columns=['user', 'message_id'])

        probs_feature_holdout = feature_model.get_model().predict_proba(feature_holdout_data)
        probs_cnn_holdout = cnn.predict_proba_batch(embeddings_holdout_data)

        X_meta_holdout = np.hstack([probs_feature_holdout, probs_cnn_holdout])
        y_meta_holdout = feature_holdout_dataset['user'].values

        df_meta_holdout = pd.DataFrame(
            X_meta_holdout,
            columns=[f"probs_feature{i}" for i in range(probs_feature_holdout.shape[1])] + 
                    [f"probs_cnn{i}" for i in range(probs_cnn_holdout.shape[1])]
        )
        df_meta_holdout['user'] = y_meta_holdout

        df_meta_holdout_enhanced = md.enhance_meta_dataset(df_meta_holdout)
        X_meta_enhanced = df_meta_holdout_enhanced.drop("user", axis=1)
        y_meta_enhanced = df_meta_holdout_enhanced['user'].values

        # Evaluate meta-learner on the holdout set
        print("\n--- Meta-Learner Performance on Holdout Set ---")
        holdout_accuracy, holdout_report, holdout_auc, holdout_confusion = meta_learner.evaluate(X_meta_enhanced, y_meta_enhanced)

        # Plot the holdout results
        print("\n--- Visualizing Holdout Set Results ---")
        meta_learner.plot_metrics(holdout_report, holdout_confusion)
