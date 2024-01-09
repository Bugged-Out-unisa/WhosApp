import argparse
from utility.logging import init_logging
from utility.model.modelTraining import ModelTraining
from utility.selection_cmdline.datasetSelection import DatasetSelection
from utility.selection_cmdline.modelSelection import ModelSelection

# HOW TO USE:
# py newTraining.py -oN <*outputName> -r <*retrain>

# CHECKS IF SPECIFIED DATASET EXIST
# (dataCreation.py return already existing DF)

# ELSE IT CREATES A NEW DATASET WITH SPECIFIED NAME from newDataset.py

# ONCE A DATASET IS GIVEN, IT TRAINS MODEL THEN PERSISTS IT


if __name__ == "__main__":
    # LOGGING:: Inizializza il logging
    init_logging("training-report.log", "!! NEW TRAINING !!")

    # Argomenti da linea di comando
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-oN", "--outputName", help="Nome file del modello salvato")
    parser.add_argument("-r", "--retrain", action="store_true", help="Opzione di retraining", required=False)

    args = parser.parse_args()
    outputName, retrain = args.outputName, args.retrain

    # Select dataset
    dataset = DatasetSelection().dataset

    # Select model
    model = ModelSelection().model

    # Training del modello con i parametri passati da linea di comando
    ModelTraining(outputName, model, dataset, retrain).run()
