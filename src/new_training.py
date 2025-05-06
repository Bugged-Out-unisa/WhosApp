import time
import calendar
import argparse
from utility.logging import LoggerReport, LoggerUserModelHistory
from utility.model.modelTraining_feature import ModelTraining
from utility.model.modelTraining_embeds import CNN1D
from utility.cmdlineManagement.datasetSelection import DatasetSelection
from utility.cmdlineManagement.modelSelection import ModelSelection

# HOW TO USE:
# py new_training.py -oN <*outputName> -c <*configFile> -st <feature|embeddings|both> -r <*retrain>

# CHECKS IF SPECIFIED DATASET EXIST
# (dataCreation.py return already existing DF)

# ELSE IT CREATES A NEW DATASET WITH SPECIFIED NAME from new_dataset.py

# ONCE A DATASET IS GIVEN, IT TRAINS MODEL THEN PERSISTS IT


if __name__ == "__main__":
    # Argomenti da linea di comando
    parser = argparse.ArgumentParser()

    timestamp = str(calendar.timegm(time.gmtime()))

    # Optional arguments
    parser.add_argument("-oN", "--outputName", help="Nome file del modello salvato" , required=False, default=timestamp)
    parser.add_argument("-c", "--config", help="File config", required=False)
    parser.add_argument("-r", "--retrain", action="store_true", help="Opzione di retraining", required=False)
    parser.add_argument("-st", "--select_training", help="selezione se eseguire training feature o embeddings", required=False, default="both")

    args = parser.parse_args()
    output_name, config, retrain, select_training = args.outputName, args.config, args.retrain, args.select_training

    feature_training = False
    embeddings_training = False

    select_training = select_training.lower()
    if select_training not in ["feature", "embeddings", "both"]:
        raise ValueError("Invalid value for --select_training. Choose 'feature', 'embeddings', or 'both'.")
    
    if select_training == "feature":
        feature_training = True
    elif select_training == "embeddings":
        embeddings_training = True
    elif select_training == "both":
        feature_training = True
        embeddings_training = True


    # LOGGING:: Inizializza il logging
    LoggerReport(
        name=output_name,
        start_message="!! NEW TRAINING !!",
        path=LoggerReport.TRAINING_LOGGING_PATH
    ).run()

    # Select dataset
    dataset_selection = DatasetSelection()
    dataset = dataset_selection.dataset
    dataset_name = dataset_selection.dataset_name

    if feature_training:
        # Select model
        model = ModelSelection().model

        LoggerUserModelHistory.append_model_user(dataset_name, output_name)

        # Training del modello con i parametri passati da linea di comando
        model_training = ModelTraining(output_name, model, dataset, config, retrain)
        model_training.run()

    if embeddings_training:
        # Training del modello con i parametri passati da linea di comando
        model = CNN1D(dataset, output_name=output_name, retrain=retrain)
        model.train_and_evaluate()