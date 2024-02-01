import os
import time
import logging
import calendar
import pandas as pd
from collections.abc import Hashable
from utility.dataset.extractChat import ExtractChat
from utility.exceptions import SetterNotAllowedError
from utility.dataset.rawDataReader import rawDataReader
from utility.clean_coding.decorator import check_path_exists
from utility.dataset.dataFrameProcess import DataFrameProcessor
from utility.dataset.featureConstruction import featureConstruction
from utility.config_path import DATASET_PATH, CONFIG_PATH, RAWDATA_PATH
from utility.clean_coding.ensure import *


@check_path_exists(path=DATASET_PATH, create=True)
@validation(
    "dataset_name",
    "Nome del dataset",
    ensure_valid_file_extension(".parquet"), ensure_not_none("dataset_" + str(calendar.timegm(time.gmtime()))))
@validation(
    "config_file",
    "File di configurazione",
    ensure_valid_file_extension(".json"), ensure_file_exists("../configs/", ""), ensure_not_none("config.json"))
@validation(
    "alias_file",
    "File per gli alias in chat",
    ensure_valid_file_extension(".json"), ensure_file_exists("../configs/", "", allow_none=True), ensure_not_none("aliases.json"))
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
    ensure_valid_type(bool), ensure_not_none(False))
class datasetCreation:
    def __init__(
            self,
            dataset_name: str,
            config_file: str,
            alias_file: str,
            other_user: str = None,
            remove_other: bool = False,
            refactor: bool = False
    ):
        self.__dataset_name = dataset_name
        self.__config_file = config_file
        self.__alias_file = alias_file

        self.__other_user = other_user if self.__alias_file is not None else None
        self.__remove_other = remove_other if self.__alias_file is not None else False

        self.__refactor = refactor
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

        if not os.path.exists(DATASET_PATH + self.__dataset_name) or self.__refactor:
            self.__create_dataset()
        elif os.path.exists(DATASET_PATH + self.__dataset_name):
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

    @staticmethod
    def __read_raw_data():
        """
        Legge i dati grezzi
        :return: dati grezzi
        """
        return rawDataReader(RAWDATA_PATH).read_all_files()

    def __extract_info(self, rawdata):
        """
        Estrae le informazioni dai dati grezzi
        :param rawdata: dati grezzi
        :return: tuple di date, utenti e messaggi
        """
        if self.__alias_file:
            return ExtractChat(
                rawdata,
                aliases=CONFIG_PATH + self.__alias_file,
                placeholder_user=self.__other_user
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
            DATASET_PATH + self.__dataset_name,
            self.__config_file
        ).get_dataframe()

    def __load_existing_dataset(self):
        """Carica un dataset esistente"""
        print("[INFO] Trovato dataset esistente")
        self.__dataFrame = pd.read_parquet(DATASET_PATH + self.__dataset_name)
