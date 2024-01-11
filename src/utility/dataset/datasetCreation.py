import os
import time
import logging
import calendar
import pandas as pd
from utility.dataset.extractChat import ExtractChat
from utility.dataset.rawDataReader import rawDataReader
from utility.dataset.dataFrameProcess import DataFrameProcessor
from utility.dataset.featureConstruction import featureConstruction


class datasetCreation:
    DATA_PATH = "../rawdata"
    DATASET_PATH = "../datasets/"
    CONFIG_PATH = "../configs/"

    def __init__(
            self,
            datasetName: str = None,
            configFile: str = "config.json",
            aliasFile: str = None,
            otherUser: str = None,
            remove_other: bool = False,
            refactor: bool = False
    ):
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
            self.__otherUser = otherUser
            self.__removeOther = remove_other
        else:
            self.__aliasFile = aliasFile
            self.__otherUser = None
            self.__removeOther = False

        # Controlla se refactor è stato inserito
        if not isinstance(refactor, bool):
            raise TypeError("refactor deve essere un booleano")
        self.__isToRefactor = True if refactor else False

        self.__dataFrame = None
        self.__check_dataset_path()

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

    def run(self):
        """Avvia la creazione del dataset."""

        # LOGGING:: Stampa il nome del dataset
        logging.info(f"Dataset name: {self.__datasetName}")

        # se il file non esiste oppure è richiesta un ricreazione di esso, esegue tutte le operazioni
        if not os.path.exists(self.DATASET_PATH + self.__datasetName) or self.__isToRefactor:

            print("\n[LOADING] Leggendo le chat dai file grezzi...")
            rawdata = rawDataReader(self.DATA_PATH).read_all_files()

            print("\n[LOADING] Estraendo informazioni dai dati grezzi...")
            if self.__aliasFile:
                dates, users, messages = ExtractChat(
                    rawdata,
                    self.CONFIG_PATH + self.__aliasFile,
                    placeholder_user=self.__otherUser,
                    remove_generic_user=self.__removeOther
                ).extract()
            else:
                dates, users, messages = ExtractChat(rawdata).extract()

            print("\n[LOADING] Creando il dataframe e applicando data cleaning e undersampling...")
            self.__dataFrame = DataFrameProcessor(dates, users, messages).get_dataframe()

            print("\n[LOADING] Applicando feature construction...")
            self.__dataFrame = featureConstruction(
                self.__dataFrame,
                self.DATASET_PATH + self.__datasetName,
                self.CONFIG_PATH + self.__configFile
            )

            print("[INFO] Dataset creato con successo.")

        # se il file esiste già allora salva il DF esistente
        elif os.path.exists(self.DATASET_PATH + self.__datasetName):
            print("[INFO] Trovato dataset esistente")
            self.__dataFrame = pd.read_parquet(self.DATASET_PATH + self.__datasetName)

    # -------- Getter --------
    @property
    def dataFrame(self):
        return self.__dataFrame
