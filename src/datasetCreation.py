import sys
import os
import argparse
import pandas as pd
from utility.featureConstruction import featureConstruction
from utility.rawDataReader import rawDataReader
from utility.extractChat import ExtractChat
from utility.data_framing import DataFrameProcessor

# HOW TO USE
# datasetCreation.py <datasetName> -c <*configFile>-r <*refactor>
    # if datasetName exists
        #if refactor is specified then create dataset with said name
        #else return already made dataset
    # else create dataset based on rawdata with that name

    # [W I P] you can use config.cfg to choose which function to run... 

class datasetCreation():

    def __init__(self, datasetName :str = None, configFile= "config.cfg",refactor :bool = False):
        self.DATA_PATH = "../rawdata"
        self.DATASET_PATH = "../datasets/"
        self.CONFIG_PATH = "../configs/"
        self.__datasetName = None
        self.__configFile= configFile

        try:
            self.__datasetName = datasetName if datasetName.endswith(".parquet") else  datasetName + ".parquet"
        except Exception:
            pass

        self.__isToRefactor = refactor
        self.__dataFrame = None

        # Crea cartella dataset se non esiste
        if not os.path.exists(self.DATASET_PATH):
            os.makedirs(self.DATASET_PATH)

        self.__main__()

    def __main__(self):
        #Se non passato per funzione controlla args da cmd line
        parser = argparse.ArgumentParser()

        # mandatory argument
        parser.add_argument("datasetName", help="Nome dataset")

        # optional arguments
        parser.add_argument("-c", "--config", help="File config", required=False)
        parser.add_argument("-r", "--refactor", help="Opzione di refactor", required=False)

        args = None

        if self.__datasetName is None:
            try:
                args = parser.parse_args()

                self.__datasetName = args.datasetName if args.datasetName.endswith(".parquet") else  args.datasetName + ".parquet"
            except:
                raise Exception("Non è stato inserito un nome per il dataset")
        
            if args.config:
                self.__configFile = args.config
            
            if args.refactor:
                self.__isToRefactor = True if args.refactor == "refactor" else False

        # se il file non esiste oppure è richiesta un ricreazione di esso, esegue tutte le operazioni
        if(not os.path.exists(self.DATASET_PATH + self.__datasetName) or self.__isToRefactor):

            print("\n[LOADING] Leggendo le chat dai file grezzi...")
            rawdata = rawDataReader(self.DATA_PATH).read_all_files()

            print("\n[LOADING] Estraendo informazioni dai dati grezzi...")
            dates, users, messages = ExtractChat(rawdata).extract()

            print("\n[LOADING] Creando il dataframe e applicando data cleaning e undersampling...")
            self.__dataFrame = DataFrameProcessor(dates, users, messages).get_dataframe()

            print("\n[LOADING] Applicando feature construction...")
            featureConstruction(self.__dataFrame, self.DATASET_PATH + self.__datasetName, self.CONFIG_PATH + self.__configFile)

            print("[INFO] Dataset creato con successo.")
        
        # se il file esiste già allora salva  il DF esistente
        elif(os.path.exists(self.DATASET_PATH + self.__datasetName)):
            print("[INFO] Trovato dataset esistente")
            self.__dataFrame = pd.read_parquet(self.DATASET_PATH + self.__datasetName)

    def getDataframe(self):
        return self.__dataFrame

if __name__ == "__main__":
    datasetCreation()