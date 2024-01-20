import os
import json


class LoggerUserModelHistory:
    """
    Classe che serve ad assocciare ad ogni modello gli utenti usati
    """

    FRONTEND_USER_PATH = "../configs/"
    LOG_ALIAS_PATH = "../logs/aliases/"

    __filename = "frontend_users.json"

    @classmethod
    def append_model_user(cls, alias_file: str, model_name: str):
        cls.__check_log_path()
        alias_file = cls.__check_alias_path(alias_file)

        with open(cls.LOG_ALIAS_PATH + alias_file, "r") as f:
            new_data = json.load(f)

        current_data = cls.__read_model_user()
        current_data[model_name] = new_data

        with open(cls.FRONTEND_USER_PATH + cls.__filename, "w") as f:
            json.dump(current_data, f, indent=4)

    @classmethod
    def __read_model_user(cls):
        cls.__check_log_path()
        if not os.path.exists(cls.FRONTEND_USER_PATH + cls.__filename):
            return {}

        with open(cls.FRONTEND_USER_PATH + cls.__filename, "r") as f:
            return json.load(f)

    @classmethod
    def __check_alias_path(cls, alias_file: str):
        alias_file = alias_file.split('.')[0] + ".json"

        if not os.path.exists(cls.LOG_ALIAS_PATH + alias_file):
            raise FileNotFoundError(f"Il file {alias_file} non esiste")

        return alias_file

    @classmethod
    def __check_log_path(cls):
        if not os.path.exists(cls.FRONTEND_USER_PATH):
            os.makedirs(cls.FRONTEND_USER_PATH)

        if not os.path.exists(cls.FRONTEND_USER_PATH + cls.__filename):
            with open(cls.FRONTEND_USER_PATH + cls.__filename, "w") as f:
                json.dump({}, f, indent=4)
