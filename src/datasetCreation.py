import os
import time
import logging
import argparse
import datetime
import calendar
import pandas as pd
from utility.extractChat import ExtractChat
from utility.rawDataReader import rawDataReader
from utility.data_framing import DataFrameProcessor
from utility.featureConstruction import featureConstruction
from utility.logging import init_logging


# HOW TO USE
# datasetCreation.py -dN <datasetName> -c <*configFile> -a <*aliases> -r <*refactor>
#   if datasetName exists
#       if refactor is specified then create dataset with said name
#       else return already made dataset
#   else create dataset based on rawdata with that name

# [W I P] you can use config.json to choose which function to run...


class datasetCreation:
    DATA_PATH = "../rawdata"
    DATASET_PATH = "../datasets/"
    CONFIG_PATH = "../configs/"

    def __init__(self, datasetName: str = None, configFile="config.json", aliasFile=None, refactor: bool = False):
        # Controlla se il nome del dataset è stato inserito
        if datasetName:
            self.__datasetName = self.__check_extension_file(datasetName, ".parquet")
        else:
            timestamp = calendar.timegm(time.gmtime())
            self.__datasetName = "dataset_" + str(timestamp) + ".parquet"

        # Controlla se configFile è stato inserito
        if configFile:
            self.__configFile = self.__check_extension_file(configFile, ".json")
        else:
            self.__configFile = "config.json"

        # Controlla se aliasFile è stato inserito
        if aliasFile:
            self.__aliasFile = self.__check_extension_file(aliasFile, ".json")
        else:
            self.__aliasFile = aliasFile

        # Controlla se refactor è stato inserito
        if not isinstance(refactor, bool):
            raise TypeError("refactor deve essere un booleano")
        self.__isToRefactor = True if refactor else False

        self.__dataFrame = None
        self.__check_dataset_path()

        # Crea il dataset
        self.__main__()

    @staticmethod
    def __check_extension_file(filename: str, ext: str):
        """Controlla se l'estensione del file è quella specificata"""
        if not filename.endswith(ext):
            filename += ext
        return filename

    @classmethod
    def __check_dataset_path(cls):
        """Crea cartella dataset se non esiste"""
        if not os.path.exists(cls.DATASET_PATH):
            os.makedirs(cls.DATASET_PATH)

    def __main__(self):

        # LOGGING:: Stampa il nome del dataset
        logging.info(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
        logging.info(f"!!NEW DATASET CREATION!! ")
        logging.info(f"Dataset name: {self.__datasetName}")

        # se il file non esiste oppure è richiesta un ricreazione di esso, esegue tutte le operazioni
        if not os.path.exists(self.DATASET_PATH + self.__datasetName) or self.__isToRefactor:

            print("\n[LOADING] Leggendo le chat dai file grezzi...")
            rawdata = rawDataReader(self.DATA_PATH).read_all_files()

            dates = users = messages = None

            print("\n[LOADING] Estraendo informazioni dai dati grezzi...")
            if self.__aliasFile:
                dates, users, messages = ExtractChat(rawdata, self.CONFIG_PATH + self.__aliasFile).extract()
            else:
                dates, users, messages = ExtractChat(rawdata).extract()

            print("\n[LOADING] Creando il dataframe e applicando data cleaning e undersampling...")
            self.__dataFrame = DataFrameProcessor(dates, users, messages).get_dataframe()

            print("\n[LOADING] Applicando feature construction...")
            featureConstruction(self.__dataFrame, self.DATASET_PATH + self.__datasetName,
                                self.CONFIG_PATH + self.__configFile)

            print("[INFO] Dataset creato con successo.")

        # se il file esiste già allora salva  il DF esistente
        elif os.path.exists(self.DATASET_PATH + self.__datasetName):
            print("[INFO] Trovato dataset esistente")
            self.__dataFrame = pd.read_parquet(self.DATASET_PATH + self.__datasetName)

    # -------- Getter --------
    @property
    def dataFrame(self):
        return self.__dataFrame


def args_cmdline():
    # Se non passato per funzione controlla args da cmd line
    parser = argparse.ArgumentParser()

    parser.add_argument("-dN", "--datasetName", help="Nome dataset", required=False)
    parser.add_argument("-c", "--config", help="File config", required=False)
    parser.add_argument("-a", "--aliases", help="File per gli alias in chat", required=False)
    parser.add_argument("-r", "--refactor", help="Opzione di refactor", action="store_true", required=False)

    args = parser.parse_args()

    return [v if v is not None else None for k, v in vars(args).items()]


if __name__ == "__main__":
    input_cmd_line = args_cmdline()
    init_logging("dataset-creation.log")
    datasetCreation(*input_cmd_line)
