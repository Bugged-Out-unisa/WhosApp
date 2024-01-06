import sys
import os
import pandas as pd
from utility.featureConstruction import featureConstruction
from utility.rawDataReader import rawDataReader
from utility.extractChat import ExtractChat
from utility.dataFrameProcess import DataFrameProcessor

# HOW TO USE
# datasetCreation.py <datasetName> <*refactor>
    # if datasetName exists
        #if refactor is specified then create dataset with said name
        #else return already made dataset
    # else create dataset based on rawdata with that name

class datasetCreation():

    def __init__(self, datasetName :str = None, refactor :bool = None):
        self.DATA_PATH = "../rawdata"
        self.DATASET_PATH = "../datasets/"
        self.__datasetName = datasetName if datasetName.endswith(".parquet") else  datasetName + ".parquet"
        self.__isToRefactor = refactor
        self.__dataFrame = None
        self.__main__()

    def __main__(self):
        if self.__datasetName is None:
            try:
                datasetName = sys.argv[1]

                if datasetName == "base":
                    self.__datasetName = "dataset.parquet"
                
                else:
                    self.__datasetName = datasetName if datasetName.endswith(".parquet") else  datasetName + ".parquet"
            except:
                raise Exception("Non è stato inserito un nome per il dataset")
        
        if self.__isToRefactor is None:
            try:
                self.__isToRefactor = True if sys.argv[2] == "refactor" else False

            except:
                self.__isToRefactor = False

        if(not os.path.exists(self.DATASET_PATH + self.__datasetName) or self.__isToRefactor):

            print("\n[LOADING] Leggendo le chat dai file grezzi...")
            rawdata = rawDataReader(self.DATA_PATH).read_all_files()

            print("\n[LOADING] Estraendo informazioni dai dati grezzi...")
            dates, users, messages = ExtractChat(rawdata).extract()

            print("\n[LOADING] Creando il dataframe e applicando data cleaning e undersampling...")
            self.__dataFrame = DataFrameProcessor(dates, users, messages).get_dataframe()

            print("\n[LOADING] Applicando feature construction...")
            featureConstruction(self.__dataFrame, self.DATASET_PATH, self.__datasetName)

            print("[INFO] Dataset creato con successo.")
        
        elif(os.path.exists(self.DATASET_PATH + self.__datasetName)):
            self.__dataFrame = pd.read_parquet(self.DATASET_PATH + self.__datasetName)

    def getDataframe(self):
        return self.__dataFrame

if __name__ == "__main__":
    datasetCreation()