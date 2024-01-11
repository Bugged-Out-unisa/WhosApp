import time
import calendar
import argparse
from utility.logging import Logger
from utility.dataset.datasetCreation import datasetCreation
from utility.cmdlineManagement.PlaceholderUserManager import PlaceholderUserManager
from utility.model.modelTraining import ModelTraining
from utility.cmdlineManagement.datasetSelection import DatasetSelection
from utility.cmdlineManagement.modelSelection import ModelSelection

"""
Per eseguire il comando da linea di comando con gli argomenti opzionali:

Comando per la creazione del dataset e il training del modello:
    py pipeline.py -oN <nome_modello> -dN <nome_dataset> -c <file_configurazione> -a <file_alias> -r -ref

Dove:
    -oN/--outputName: Specifica il nome del file per il modello salvato.
    -dN/--datasetName: Specifica il nome del dataset.
    -c/--config: Specifica il file di configurazione per la creazione del dataset.
    -a/--aliases: Specifica il file degli alias per l'utente "other".
    -r/--retrain: Opzione per indicare il retraining del modello (opzionale).
    -ref/--refactor: Opzione per il refactor (opzionale).

Esempio di utilizzo:
    python nome_script.py -oN modello1 -r -dN dataset1 -c config.json -a aliases.txt -ref
"""


def create_dataset_and_train_model():

    parser = argparse.ArgumentParser()

    # Argomenti comuni a entrambi gli script
    parser.add_argument("-oN", "--outputName", help="Nome file del modello salvato")
    parser.add_argument("-r", "--retrain", action="store_true", help="Opzione di retraining", required=False)
    parser.add_argument("-dN", "--datasetName", help="Nome dataset", required=False)
    parser.add_argument("-c", "--config", help="File config", required=False)
    parser.add_argument("-a", "--aliases", help="File per gli alias in chat", required=False)
    parser.add_argument("-ref", "--refactor", help="Opzione di refactor", action="store_true", required=False)

    args = parser.parse_args()

    # Parametri comuni
    output_name, retrain = args.outputName, args.retrain
    dataset_name, config, aliases_file, refactor = args.datasetName, args.config, args.aliases, args.refactor

    # Selezione opzioni per l'utente "other"
    placeholder_user, remove_generic = PlaceholderUserManager(aliases_file).selection()

    # Selezione del modello
    selected_model = ModelSelection().model

    # Definizione del timestamp
    timestamp = str(calendar.timegm(time.gmtime()))

    # Se il dataset_name non è None, lo imposta al timestamp. Altrimenti lo usa
    dataset_name = dataset_name if dataset_name is not None else timestamp

    # LOGGING:: Inizializza il logging
    Logger(
        name=dataset_name,
        path=Logger.DATASET_LOGGING_PATH,
        start_message="!! NEW DATASET CREATION !!"
    ).run()

    # Creazione del dataset con i parametri passati da linea di comando
    dataset_creator = datasetCreation(
        dataset_name,
        config,
        aliases_file,
        placeholder_user,
        remove_generic,
        refactor
    )

    dataset_creator.run()

    # Usa il dataset creato oppure effettua la selezione del dataset
    selected_dataset = dataset_creator.dataFrame if dataset_creator.dataFrame is not None else DatasetSelection().dataset

    # Se il output_name non è None, lo imposta al timestamp. Altrimenti lo usa
    output_name = args.outputName if args.outputName is not None else timestamp

    # LOGGING:: Re-inizializza il logging
    Logger(
        name=output_name,
        start_message="!! NEW TRAINING !!",
        path=Logger.TRAINING_LOGGING_PATH
    ).run()

    # Training del modello con i parametri passati da linea di comando
    ModelTraining(output_name, selected_model, selected_dataset, retrain).run()


if __name__ == "__main__":
    create_dataset_and_train_model()
