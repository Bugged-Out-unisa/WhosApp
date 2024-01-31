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
from utility.decorator import check_file_exists, check_extension_file, check_type_param


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
        self.__dataset_name = self.__check_dataset_name(dataset_name)
        self.__config_file = self.__check_config_file(config_file)
        self.__alias_file = self.__check_aliases_file(alias_file)

        self.__other_user = self.__check_other_user(other_user)
        self.__remove_other = self.__check_remove_other(remove_other)

        self.__isToRefactor = self.__check_refactor(refactor)
        self.__dataFrame = None

    # -------- Controlli --------

    @staticmethod
    @check_file_exists(base_path=CONFIG_PATH, subdir="")
    @check_extension_file(".json")
    def __check_config_file(config_file: str) -> str:
        return config_file if config_file is not None else "config.json"

    @staticmethod
    @check_file_exists(base_path=CONFIG_PATH, subdir="", allow_none=True)
    @check_extension_file(".json")
    def __check_aliases_file(aliases_file: str):
        return aliases_file if aliases_file is not None else None

    @staticmethod
    @check_extension_file(".parquet")
    def __check_dataset_name(name: str) -> str:
        return name if name is not None else "dataset_" + str(calendar.timegm(time.gmtime())) + ".parquet"

    @check_type_param("other_user", Hashable, allow_none=True)
    def __check_other_user(self, other_user: str):
        return other_user if self.__alias_file is not None else None

    @check_type_param("remove_other", bool)
    def __check_remove_other(self, remove_other: bool):
        return remove_other if self.__alias_file is not None else False

    @staticmethod
    @check_type_param("refactor", bool)
    def __check_refactor(refactor: bool):
        return refactor if refactor is not None else refactor

    # -------- Getter & Setter --------

    @property
    def data_frame(self):
        return self.__dataFrame

    @data_frame.setter
    def data_frame(self, data_frame):
        raise SetterNotAllowedError("DataFrame must be generated through 'run' method")

    # -------- Main Method --------

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
        return DataFrameProcessor(dates, users, messages, self.__other_user, self.__remove_other).get_dataframe()

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


