import os
import json


class LoggerUser:
    """
    Classe che serve a memorizzare gli alias degli utenti in fase di Dataset Creation
    """

    FRONTEND_USER_PATH = "../logs/aliases/"
    __filename = None
    __file_pointer = None

    @classmethod
    def __check_log_path(cls):
        if not os.path.exists(cls.FRONTEND_USER_PATH):
            os.makedirs(cls.FRONTEND_USER_PATH)

    @classmethod
    def __check_filename(cls, filename: str):
        if filename is not None:
            if not filename.endswith(".json"):
                filename += ".json"
            cls.__filename = filename
        else:
            raise ValueError("Il nome del file non può essere nullo")

    @classmethod
    def open(cls, filename: str = __filename):
        cls.__check_filename(filename)
        cls.__check_log_path()
        cls.__open_file()
        return cls.__file_pointer

    @classmethod
    def __open_file(cls):
        try:
            cls.__file_pointer = open(cls.FRONTEND_USER_PATH + cls.__filename, "w")
        except IOError:
            raise IOError(f"Impossibile aprire il file {cls.__filename}")

    @classmethod
    def close(cls):
        if cls.__file_pointer is not None:
            cls.__file_pointer.close()

    @classmethod
    def write_user(cls, user: list, index: range):
        data = {}
        for i, u in zip(index, user):
            data[i] = u

        print(data)
        cls.__write(data)

    @classmethod
    def __write(cls, data: dict):
        cls.__check_log_path()
        json.dump(data, cls.__file_pointer, indent=4)
        cls.close()