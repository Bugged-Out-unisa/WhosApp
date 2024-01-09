import os
import logging

LOGGING_PATH = "../logs/"


def init_logging(filename: str = "log.log"):
    if not os.path.exists(LOGGING_PATH):
        os.makedirs(LOGGING_PATH)

    filename = filename if filename.endswith(".log") else filename + ".log"

    logging.basicConfig(
        filename=LOGGING_PATH + filename,
        level=logging.INFO,
        format='%(message)s'
    )
