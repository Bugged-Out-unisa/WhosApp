import re
from datetime import datetime
from tqdm import tqdm


class ExtractChat:
    REGEX_TIMESTAMP_BASE = r"\d{1,2}/\d{1,2}/\d{2,4}, \d{2}:\d{2}"
    REGEX_TIMESTAMP_FOR_ANDROID = REGEX_TIMESTAMP_BASE + r" - "
    REGEX_TIMESTAMP_FOR_IOS = r"\[" + REGEX_TIMESTAMP_BASE + r":\d{2}\] "

    def __init__(self, rawdata: str):
        self.__rawdata = rawdata
        self.__regex_timestamp = None

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
                if (len(numbers) == 6): numbers = numbers[:-1]

                # Estrai valori singoli
                day, month, year, hour, minute  = numbers

                # Espandi il formato dell'anno (yy -> yyyy)
                if (len(str(year)) == 2): year += 2000

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

        return dates, users, messages