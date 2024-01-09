import argparse
from utility.logging import init_logging
from utility.dataset.datasetCreation import datasetCreation


# HOW TO USE
# py newDataset.py -dN <*datasetName> -c <*configFile> -a <*aliases> -r <*refactor>
#   if datasetName exists
#       if refactor is specified then create dataset with said name
#       else return already made dataset
#   else create dataset based on rawdata with that name

# [W I P] you can use config.json to choose which function to run...


if __name__ == "__main__":
    # LOGGING:: Inizializza il logging
    init_logging("dataset-report.log", "!! NEW DATASET CREATION !!")

    # Argomenti da linea di comando
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-dN", "--datasetName", help="Nome dataset", required=False)
    parser.add_argument("-c", "--config", help="File config", required=False)
    parser.add_argument("-a", "--aliases", help="File per gli alias in chat", required=False)
    parser.add_argument("-r", "--refactor", help="Opzione di refactor", action="store_true", required=False)

    args = parser.parse_args()
    dataset_name, config, aliases, refactor = args.datasetName, args.config, args.aliases, args.refactor

    # Creazione del dataset con i parametri passati da linea di comando
    datasetCreation(dataset_name, config, aliases, refactor).run()
