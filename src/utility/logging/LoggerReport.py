import os
import logging
import datetime


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
            path: str = DEFAULT_LOGGING_PATH
    ):
        self.__name = name if name is not None else "unknown"
        self.__start_message = start_message if start_message is not None else "!! START NEW LOG !!"
        self.__path = path if path in self.__logging_paths else self.DEFAULT_LOGGING_PATH
        self.__filelog = None

        # Controlla se le path esistono
        self.__check_path()
        self.__check_path(self.__path)

        # Crea il file di log
        self.__check_filelog()

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name
        self.__check_filelog()

    @staticmethod
    def __check_path(path: str = DEFAULT_LOGGING_PATH):
        if not os.path.exists(path):
            os.makedirs(path)

    def __check_filelog(self):
        if not self.__name.endswith(".log"):
            self.__name += ".log"

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
