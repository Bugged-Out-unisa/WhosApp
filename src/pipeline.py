import time
import calendar
import argparse
from utility.logging import LoggerReport, LoggerUser, LoggerUserModelHistory
from utility.model.modelTraining_feature import ModelTraining
from utility.model.modelTraining_embeddings import CNN1D
from utility.dataset.datasetCreation import datasetCreation
from utility.cmdlineManagement.modelSelection import ModelSelection
from utility.cmdlineManagement.datasetSelection import DatasetSelection
from utility.cmdlineManagement.PlaceholderUserManager import PlaceholderUserManager

"""
Per eseguire il comando da linea di comando con gli argomenti opzionali:

Comando per la creazione del dataset e il training del modello:
    py pipeline.py -oN <nome_modello_dataset> -c <file_configurazione> -a <file_alias> -r -ref

Dove:
    -oN/--outputName: Specifica il nome del file per il modello salvato e del modello.
    -c/--config: Specifica il file di configurazione per la creazione del dataset.
    -a/--aliases: Specifica il file degli alias per l'utente "other".
    -r/--retrain: Opzione per indicare il retraining del modello (opzionale).
    -ref/--refactor: Opzione per il refactor (opzionale).

Esempio di utilizzo:
    python nome_script.py -oN modello1 -r -dN dataset1 -c config.json -a aliases.txt -ref
"""


def create_dataset_and_train_model():
    parser = argparse.ArgumentParser()

    # Definizione del timestamp
    timestamp = str(calendar.timegm(time.gmtime()))

    # Argomenti comuni a entrambi gli script
    parser.add_argument("-oN", "--outputName", help="Nome del dataset e del modello", required=False, default=timestamp)
    parser.add_argument("-r", "--retrain", action="store_true", help="Opzione di retraining", required=False)
    parser.add_argument("-c", "--config", help="File config", required=False)
    parser.add_argument("-a", "--aliases", help="File per gli alias in chat", required=False)
    parser.add_argument("-ref", "--refactor", help="Opzione di refactor", action="store_true", required=False)
    parser.add_argument("-st", "--select_training", help="selezione se eseguire training feature o embeddings", required=False, default="both")

    args = parser.parse_args()

    # Parametri comuni
    output_name, retrain = args.outputName, args.retrain
    dataset_name, config, aliases_file, refactor, select_training = args.outputName, args.config, args.aliases, args.refactor, args.select_training

    feature_training = False
    embeddings_training = False

    if select_training not in ["feature", "embeddings", "both"]:
        raise ValueError("Invalid value for --select_training. Choose 'feature', 'embeddings', or 'both'.")
    
    if select_training == "feature":
        feature_training = True
    elif select_training == "embeddings":
        embeddings_training = True
    elif select_training == "both":
        feature_training = True
        embeddings_training = True

    # Selezione opzioni per l'utente "other"    
    placeholder_user, remove_generic = PlaceholderUserManager(aliases_file).selection()

    # LOGGING:: Inizializza il logging
    LoggerReport(
        name=output_name,
        start_message="!! START NEW PIPELINE !!",
        path=LoggerReport.PIPELINE_LOGGING_PATH
    ).run()

    LoggerUser.open(output_name)

    # Selezione del modello
    selected_model = ModelSelection().model

    # Creazione del dataset con i parametri passati da linea di comando
    dataset_creator = datasetCreation(
        dataset_name,
        config,
        aliases_file,
        placeholder_user,
        remove_generic,
        refactor,
        feature_training,
        embeddings_training
    )
    dataset_creator.run()

    LoggerUserModelHistory.append_model_user(dataset_name, output_name)

    # Se il output_name non è None, lo imposta al timestamp. Altrimenti lo usa
    output_name = args.outputName if args.outputName is not None else timestamp

    if feature_training:
        # Usa il dataset creato oppure effettua la selezione del dataset
        selected_dataset = dataset_creator.dataFrame if dataset_creator.dataFrame is not None else DatasetSelection().dataset

         # Training del modello con i parametri passati da linea di comando
        ModelTraining(output_name, selected_model, selected_dataset, config, retrain).run()

    if embeddings_training:
        # Usa il dataset creato oppure effettua la selezione del dataset
        selected_embeddings = dataset_creator.embeddings_dataframe if dataset_creator.embeddings_dataframe is not None else DatasetSelection().dataset

        CNN1D(
            dataset=selected_embeddings,
            output_name=output_name,
            retrain=retrain
        ).train_and_evaluate()

   

    LoggerUser.close()


if __name__ == "__main__":
    create_dataset_and_train_model()
