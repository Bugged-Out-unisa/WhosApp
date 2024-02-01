import os
import logging
import datetime
from utility.config_path import LO
from utility.clean_coding.decorator import check_path_exists
from utility.clean_coding.ensure import validation, ensure_valid_file_extension


@check_path_exists("../logs/", create=True)
@check_path_exists("../logs/dataset/", create=True)
@check_path_exists("../logs/training", create=True)
@check_path_exists("../logs/pipeline", create=True)
@validation(
    "name",
    "Nome del file di log",
    ensure_valid_file_extension(".log")
)
class LoggerReport:
    """
    Classe che gestisce i logger statici del programma
    """

    DEFAULT_LOGGING_PATH = "../logs/"
    DATASET_LOGGING_PATH = DEFAULT_LOGGING_PATH + "dataset/"
    TRAINING_LOGGING_PATH = DEFAULT_LOGGING_PATH + "training/"
    PIPELINE_LOGGING_PATH = DEFAULT_LOGGING_PATH + "pipeline/"
    __logging_paths = (DATASET_LOGGING_PATH, TRAINING_LOGGING_PATH, PIPELINE_LOGGING_PATH)

    def __init__(
            self,
            name: str = "unknown",
            start_message: str = "!! START NEW LOG !!",
            path: str = LOGS_PATH
    ):
        self.__name = name if name is not None else "unknown"
        self.__start_message = start_message if start_message is not None else "!! START NEW LOG !!"
        self.__path = path if path in self.__logging_paths else LOGS_PATH
        self.__filelog = None

        # Formatta il nome del file di log
        self.__format_name_filelog()

    def __format_name_filelog(self):
        if self.__path == self.DATASET_LOGGING_PATH:
            name = f"report-dataset_{self.__name}"
        elif self.__path == self.TRAINING_LOGGING_PATH:
            name = f"report-training_{self.__name}"
        else:
            name = f"report-{self.__name}"

        self.__filelog = self.__path + name

    def run(self):
        logging.basicConfig(
            filename=self.__filelog,
            level=logging.INFO,
            format='%(message)s'
        )

        logging.info(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
        logging.info(self.__start_message)
