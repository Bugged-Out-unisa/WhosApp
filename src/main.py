import argparse
from utility.logging import init_logging
from utility.dataset.datasetCreation import datasetCreation
from utility.cmdlineManagement.PlaceholderUserManager import PlaceholderUserManager
from utility.model.modelTraining import ModelTraining
from utility.cmdlineManagement.datasetSelection import DatasetSelection
from utility.cmdlineManagement.modelSelection import ModelSelection

"""
Per eseguire il comando da linea di comando con gli argomenti opzionali:

Comando per la creazione del dataset e il training del modello:
    py main.py -oN <nome_modello> -dN <nome_dataset> -c <file_configurazione> -a <file_alias> -r -ref

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
    # LOGGING:: Inizializza il logging
    init_logging("combined-report.log", "!! COMBINED PROCESS !!")

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

    # Creazione del dataset con i parametri passati da linea di comando
    dataframe = datasetCreation(
        dataset_name,
        config,
        aliases_file,
        placeholder_user,
        remove_generic,
        refactor
    ).run()

    # Selezione del dataset
    selected_dataset = dataframe if dataframe is not None else DatasetSelection().dataset

    # Selezione del modello
    selected_model = ModelSelection().model

    # Training del modello con i parametri passati da linea di comando
    ModelTraining(output_name, selected_model, selected_dataset, retrain).run()


if __name__ == "__main__":
    create_dataset_and_train_model()
