import pandas as pd
import numpy as np
import skops.io as skio
from modelTraining import ModelTraining
from utility.featureConstruction import featureConstruction

class modelExecution:
    MODEL_PATH = "../models/"

    def __init__(self, model = None):
        fileModel = self.MODEL_PATH + model

        self.__trainedModel = skio.load(fileModel)
    
    def dataframe_for_messages(self, messages):
        df = pd.DataFrame({
                "date": "9/11/2001",
                "responsiveness": 600,
                "message": messages
            })
            
        return featureConstruction(dataFrame= df, datasetPath="./", createDataFrame=False).get_dataframe()

    def __predict__(self):
        #user_messages = []
        probabilities = []

        while True:
            user_messages = []
            user_messages.append(input("Scrivi un messaggio:\n"))

            df = self.dataframe_for_messages(user_messages)

            predictedUser = self.__trainedModel.predict_proba(df)
            
            for i, _ in enumerate(predictedUser[0]):
                try:
                    probabilities[i][0]
                except:
                    probabilities.append(list())
                    
            for i, value in enumerate(predictedUser[0]):
                probabilities[i].append(value)

            means = []
            for user_prob in probabilities:
                means.append(np.average(user_prob))
            
            print("----SINGOLO----\nUSER 0: {} \nUSER 1: {}".format(probabilities[0][-1], probabilities[1][-1]))
            print("-----MEDIA-----\nUSER 0: {} \nUSER 1: {}".format(means[0], means[1]))

            



if __name__ == "__main__":
    modelExecution("primo.skops").__predict__()

# user_messages = []
# user_messages.append(input("Scrivi un messaggio:\n"))

# df = self.dataframe_for_messages(user_messages)

# predictedUser = self.__trainedModel.predict_proba(df)

# print(predictedUser)