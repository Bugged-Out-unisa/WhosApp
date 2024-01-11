import time
import calendar
import argparse
from utility.logging import Logger
from utility.model.modelTraining import ModelTraining
from utility.cmdlineManagement.datasetSelection import DatasetSelection
from utility.cmdlineManagement.modelSelection import ModelSelection

# HOW TO USE:
# py new_training.py -oN <*outputName> -r <*retrain>

# CHECKS IF SPECIFIED DATASET EXIST
# (dataCreation.py return already existing DF)

# ELSE IT CREATES A NEW DATASET WITH SPECIFIED NAME from new_dataset.py

# ONCE A DATASET IS GIVEN, IT TRAINS MODEL THEN PERSISTS IT


if __name__ == "__main__":
    # Argomenti da linea di comando
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-oN", "--outputName", help="Nome file del modello salvato")
    parser.add_argument("-r", "--retrain", action="store_true", help="Opzione di retraining", required=False)

    args = parser.parse_args()
    output_name, retrain = args.outputName, args.retrain

    # Se il output_name non è None, lo imposta al timestamp. Altrimenti lo usa
    output_name = args.outputName if args.outputName is not None else str(calendar.timegm(time.gmtime()))

    # LOGGING:: Inizializza il logging
    Logger(
        name=output_name,
        start_message="!! NEW TRAINING !!",
        path=Logger.TRAINING_LOGGING_PATH
    ).run()

    # Select dataset
    dataset = DatasetSelection().dataset

    # Select model
    model = ModelSelection().model

    # Training del modello con i parametri passati da linea di comando
    ModelTraining(output_name, model, dataset, retrain).run()
