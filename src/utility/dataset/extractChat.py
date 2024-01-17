import re
import json
import logging
from tqdm import tqdm
from datetime import datetime
from utility.cmdlineManagement.PlaceholderUserManager import PlaceholderUserManager as Phum


class ExtractChat:
    REGEX_TIMESTAMP_BASE = r"\d{1,2}/\d{1,2}/\d{2,4}, \d{2}:\d{2}"
    REGEX_TIMESTAMP_FOR_ANDROID = REGEX_TIMESTAMP_BASE + r" - "
    REGEX_TIMESTAMP_FOR_IOS = r"\[" + REGEX_TIMESTAMP_BASE + r":\d{2}\] "

    def __init__(
            self,
            rawdata: str,
            aliases: str = None,
            placeholder_user: str = Phum.DEFAULT_PLACEHOLDER,
    ):
        self.__rawdata = rawdata
        self.__aliasesPath = aliases
        self.__userDict = dict()
        self.__regex_timestamp = None
        self.__placeholder_user = placeholder_user
        self.__loadAliases()

    def __loadAliases(self):
        try:
            f = open(self.__aliasesPath, "r", encoding="utf8")
        except Exception:
            return

        data = json.load(f)

        for name in data:
            for value in data[name]:
                self.__userDict[value] = name

        f.close()

    def __set_datatime(self, file):
        """
        Imposta il formato della data in base al formato del timestamp.
        """

        # Ottieni la prima riga della chat
        first_line = file[:file.find("\n")]

        # Verifica date format usato nella chat (basandosi sulla prima riga)
        if re.match(ExtractChat.REGEX_TIMESTAMP_FOR_IOS, first_line):
            self.__regex_timestamp = ExtractChat.REGEX_TIMESTAMP_FOR_IOS

        elif re.match(ExtractChat.REGEX_TIMESTAMP_FOR_ANDROID, first_line):
            self.__regex_timestamp = ExtractChat.REGEX_TIMESTAMP_FOR_ANDROID

        else:
            raise Exception("Format not supported")

    def extract(self):
        """
        Estrae le informazioni dal file di testo.
        """
        users = []
        messages = []
        dates = []

        for file in tqdm(self.__rawdata):
            # Ottieni regex adeguata
            self.__set_datatime(file)

            for match in re.findall(self.__regex_timestamp, file):
                # Estrai lista di numeri contenuti nella data "grezza"
                numbers = [int(num) for num in re.findall(r'\d+', match)]

                # Rimuovi i secondi presenti in IOS per omologarsi ad Android
                if len(numbers) == 6: numbers = numbers[:-1]

                # Estrai valori singoli
                day, month, year, hour, minute = numbers

                # Espandi il formato dell'anno (yy -> yyyy)
                if len(str(year)) == 2: year += 2000

                # Costruisci data e ottieni il suo timestamp
                dates.append(int(datetime(year, month, day, hour, minute).timestamp()))

            users_messages = re.split(self.__regex_timestamp, file)[1:]

            for message in users_messages:
                entry = re.split(r'([\w\W]+?):\s', message)

                if entry[1:]:
                    users.append(entry[1])
                    messages.append(entry[2].replace("\n", " ").strip())
                else:
                    users.append('info')
                    messages.append(entry[0])

        if (len(self.__userDict)) >= 1:
            users = [self.__userDict.get(name, self.__placeholder_user) for name in users]

        return dates, users, messages
