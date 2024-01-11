import os
import logging
import datetime

LOGGING_PATH = "../logs/"


def init_logging(filename: str = "log.log", start_message: str = None):
    if not os.path.exists(LOGGING_PATH):
        os.makedirs(LOGGING_PATH)

    filename = filename if filename.endswith(".log") else filename + ".log"

    logging.basicConfig(
        filename=LOGGING_PATH + filename,
        level=logging.INFO,
        format='%(message)s'
    )

    # Controllo del messaggio di log
    message = start_message if start_message and isinstance(start_message, str) \
        else "Avvio programma"

    logging.info(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
    logging.info(message)
