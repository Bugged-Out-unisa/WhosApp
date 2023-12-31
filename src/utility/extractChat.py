import re
from datetime import datetime
from tqdm import tqdm


class ExtractChat:
    REGEX_TIMESTAMP_FOR_XIAOMI = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    REGEX_TIMESTAMP_FOR_ANDROID = r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2} -"
    REGEX_TIMESTAMP_FOR_IOS = r"\[\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2}\]"

    FORMAT_DATE_FOR_XIAOMI = "%m/%d/%y, %H:%M - "
    FORMAT_DATE_FOR_ANDROID = "%d/%m/%Y, %H:%M -"
    FORMAT_DATE_FOR_IOS = "[%d/%m/%y, %H:%M:%S]"

    def __init__(self, rawdata: str):
        self.__rawdata = rawdata
        self.__formatdate = None
        self.__regex_timestamp = None
        self.__set_datatime()

    def __set_datatime(self):

        if re.match(ExtractChat.REGEX_TIMESTAMP_FOR_XIAOMI, self.__rawdata):
            self.__formatdate = ExtractChat.FORMAT_DATE_FOR_XIAOMI
            self.__regex_timestamp = ExtractChat.REGEX_TIMESTAMP_FOR_XIAOMI

        elif re.match(ExtractChat.REGEX_TIMESTAMP_FOR_IOS, self.__rawdata):
            self.__formatdate = ExtractChat.FORMAT_DATE_FOR_IOS
            self.__regex_timestamp = ExtractChat.REGEX_TIMESTAMP_FOR_IOS

        elif re.match(ExtractChat.REGEX_TIMESTAMP_FOR_ANDROID, self.__rawdata):
            self.__formatdate = ExtractChat.FORMAT_DATE_FOR_ANDROID
            self.__regex_timestamp = ExtractChat.REGEX_TIMESTAMP_FOR_ANDROID

        else:
            raise Exception("Format not supported")

    def extract(self):
        dates = [int(datetime.strptime(d, self.__formatdate).timestamp())
                 for d in re.findall(self.__regex_timestamp, self.__rawdata)]

        users_messages = re.split(self.__regex_timestamp, self.__rawdata)[1:]

        users = []
        messages = []
        for message in tqdm(users_messages):
            entry = re.split(r'([\w\W]+?):\s', message)

            if entry[1:]:
                users.append(entry[1])
                messages.append(entry[2].rstrip('\n'))
            else:
                users.append('info')
                messages.append(entry[0])

        return dates, users, messages

    def __enter__(self):
        return self.extract()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
