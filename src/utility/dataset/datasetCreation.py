import os
import time
import logging
import calendar
import pandas as pd
from collections.abc import Hashable
from utility.exceptions import SetterNotAllowedError
from utility.dataset.extractChat import ExtractChat
from utility.dataset.rawDataReader import rawDataReader
from utility.dataset.dataFrameProcess import DataFrameProcessor
from utility.dataset.featureConstruction import featureConstruction
from utility.clean_coding.decorator import check_path_exists
from utility.clean_coding.ensure import validation, ensure_valid_file_extension, ensure_file_exists, ensure_valid_type


@check_path_exists(path="../data/datasets/")
@validation(
    "dataset_name",
    "Nome del dataset",
    ensure_valid_file_extension(".parquet"))
@validation(
    "config_file",
    "File di configurazione",
    ensure_valid_file_extension(".json"), ensure_file_exists("../configs/", ""))
@validation(
    "alias_file",
    "File per gli alias in chat",
    ensure_valid_file_extension(".json"), ensure_file_exists("../configs/", "", allow_none=True))
@validation(
    "other_user",
    "Utente Blob",
    ensure_valid_type(Hashable, allow_none=True))
@validation(
    "remove_other",
    "Rimuovi Blob",
    ensure_valid_type(bool))
@validation(
    "refactor",
    "Opzione di refactor",
    ensure_valid_type(bool))
class datasetCreation:
    DATA_PATH = "../data/rawdata"
    DATASET_PATH = "../data/datasets/"
    CONFIG_PATH = "../configs/"

    def __init__(
            self,
            dataset_name: str,
            config_file: str,
            alias_file: str,
            other_user: str = None,
            remove_other: bool = False,
            refactor: bool = False
    ):
        self.__dataset_name = dataset_name if dataset_name is not None else "dataset_" + str(calendar.timegm(time.gmtime()))
        self.__config_file = config_file if config_file is not None else "config.json"
        self.__alias_file = alias_file if alias_file is not None else None

        self.__other_user = other_user if self.__alias_file is not None else None
        self.__remove_other = remove_other if self.__alias_file is not None else False

        self.__refactor = refactor if refactor is not None else False
        self.__data_frame = None

    # -------- Getter & Setter --------
    @property
    def data_frame(self):
        return self.__data_frame

    @data_frame.setter
    def data_frame(self, value):
        raise SetterNotAllowedError("data_frame must not be set. Use run() method instead")

    # -------- Methods --------

    def run(self):
        """Genera il frame"""
        logging.info(f"Dataset name: {self.__dataset_name}")

        if not os.path.exists(self.DATASET_PATH + self.__dataset_name) or self.__isToRefactor:
            self.__create_dataset()
        elif os.path.exists(self.DATASET_PATH + self.__dataset_name):
            self.__load_existing_dataset()

    def __create_dataset(self):
        """Crea il dataset"""
        print("\n[LOADING] Leggendo le chat dai file grezzi...")
        rawdata = self.__read_raw_data()

        print("\n[LOADING] Estraendo informazioni dai dati grezzi...")
        dates, users, messages = self.__extract_info(rawdata)

        print("\n[LOADING] Creando il dataframe e applicando data cleaning e undersampling...")
        self.__dataFrame = self.__process_data(dates, users, messages)

        print("\n[LOADING] Applicando feature construction...")
        self.__dataFrame = self.__construct_features()

        print("[INFO] Dataset creato con successo.")

    def __read_raw_data(self):
        """
        Legge i dati grezzi
        :return: dati grezzi
        """
        return rawDataReader(self.DATA_PATH).read_all_files()

    def __extract_info(self, rawdata):
        """
        Estrae le informazioni dai dati grezzi
        :param rawdata: dati grezzi
        :return: tuple di date, utenti e messaggi
        """
        if self.__alias_file:
            return ExtractChat(
                rawdata,
                self.CONFIG_PATH + self.__alias_file,
                self.__other_user
            ).extract()
        else:
            return ExtractChat(rawdata).extract()

    def __process_data(self, dates, users, messages):
        """
        Crea il dataframe e applica data cleaning e undersampling
        :param dates: date dei messaggi
        :param users: autori dei messaggi
        :param messages: messaggi
        :return: dataframe
        """
        return DataFrameProcessor(dates, users, messages, self.__other_user, self.__remove_other).run()

    def __construct_features(self):
        """
        Crea le feature del dataframe utili alla predizione
        :return: dataframe
        """

        return featureConstruction(
            self.__dataFrame,
            self.DATASET_PATH + self.__dataset_name,
            self.CONFIG_PATH + self.__config_file
        ).get_dataframe()

    def __load_existing_dataset(self):
        """Carica un dataset esistente"""
        print("[INFO] Trovato dataset esistente")
        self.__dataFrame = pd.read_parquet(self.DATASET_PATH + self.__dataset_name)
