import time
import calendar
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utility.logging import LoggerReport, LoggerUserModelHistory
from utility.model.modelTraining_feature import ModelTraining
from utility.model.modelTraining_embeddings import CNN1D, FocalLoss
from utility.model.modelTraining_meta import MetaLearner
from utility.cmdlineManagement.datasetSelection import DatasetSelection
from utility.cmdlineManagement.modelSelection import ModelSelection


# HOW TO USE:
# py new_training.py -oN <*outputName> -c <*configFile> -st <feature|embeddings|both> -r <*retrain>

# CHECKS IF SPECIFIED DATASET EXIST
# (dataCreation.py return already existing DF)

# ELSE IT CREATES A NEW DATASET WITH SPECIFIED NAME from new_dataset.py

# ONCE A DATASET IS GIVEN, IT TRAINS MODEL THEN PERSISTS IT

def slice_by_ids(df, ids):
    return df[df['message_id'].isin(ids)].reset_index(drop=True)

if __name__ == "__main__":
    # Argomenti da linea di comando
    parser = argparse.ArgumentParser()

    timestamp = str(calendar.timegm(time.gmtime()))

    # Optional arguments
    parser.add_argument("-oN", "--outputName", help="Nome file del modello salvato" , required=False, default=timestamp)
    parser.add_argument("-c", "--config", help="File config", required=False)
    parser.add_argument("-r", "--retrain", action="store_true", help="Opzione di retraining", required=False)
    parser.add_argument("-st", "--select_training", help="selezione se eseguire training feature o embeddings", required=False, default="meta")

    args = parser.parse_args()
    output_name, config, retrain, select_training = args.outputName, args.config, args.retrain, args.select_training

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

    if feature_training:
        # Select dataset
        print("-- Features --")
        dataset_selection = DatasetSelection()
        feature_dataset = dataset_selection.dataset
        feature_dataset_name = dataset_selection.dataset_name
    
    if embeddings_training:
        # Select dataset
        print("-- Embeddings --")
        dataset_selection = DatasetSelection()
        embeddings_dataset = dataset_selection.dataset
        embeddings_dataset_name = dataset_selection.dataset_name

    if meta_training:
        # stratify by 'user' so label proportions stay roughly equal
        train_ids, holdout_ids = train_test_split(
            feature_dataset['message_id'],
            test_size=0.2,
            random_state=42,
            stratify=feature_dataset['user']
        )

        # Slice both DataFrames by IDs
        feature_train_dataset   = feature_dataset[feature_dataset['message_id'].isin(train_ids)].reset_index(drop=True)
        feature_holdout_dataset = feature_dataset[feature_dataset['message_id'].isin(holdout_ids)].reset_index(drop=True)

        embeddings_train_dataset   = embeddings_dataset[embeddings_dataset['message_id'].isin(train_ids)].reset_index(drop=True)
        embeddings_holdout_dataset = embeddings_dataset[embeddings_dataset['message_id'].isin(holdout_ids)].reset_index(drop=True)

    

    if feature_training:
        # Select model
        model_choice = ModelSelection().model
        feature_train_dataset.drop("message_id", axis=1, inplace=True)
        # LoggerUserModelHistory.append_model_user(dataset_name, output_name)

        # Training del modello con i parametri passati da linea di comando
        feature_model = ModelTraining(output_name, model_choice, feature_train_dataset, config, retrain)
        feature_model.run()

    if embeddings_training:
        # Training del modello con i parametri passati da linea di comando
        embeddings_train_dataset.drop("message_id", axis=1, inplace=True)
        cnn = CNN1D(embeddings_train_dataset, output_name=output_name, retrain=retrain,)
        cnn.train_and_evaluate(criterion=FocalLoss(alpha=.5, gamma=4))

    if meta_training:
        probs_feature = feature_model.get_model().predict_proba( embeddings_holdout_dataset.drop(columns=['user','message_id']) )
        probs_cnn = cnn.predict_proba( feature_holdout_dataset.drop(columns=['user','message_id']) )

        X_meta = np.hstack([probs_feature, probs_cnn])
        y_meta = feature_holdout_dataset['user'].values

        df_meta = pd.DataFrame(
            X_meta,
            columns=[f"probs_feature{i}" for i in range(probs_feature.shape[1])]
                    +[f"probs_cnn{i}" for i in range(probs_cnn.shape[1])]
        )
        df_meta['user'] = y_meta

        meta_learner = MetaLearner(df_meta, output_name=output_name, retrain=retrain)
        meta_learner.train_and_evaluate()
