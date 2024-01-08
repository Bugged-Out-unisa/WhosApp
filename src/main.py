import pandas as pd
from modelTraining import ModelTraining
from utility.featureConstruction import featureConstruction

class modelExecution:
    MODEL_PATH = "../models/"

    def __init__(self, model = None):
        self.__trainedModel = model
    
    def dataframe_for_messages(self, messages):
        df = pd.DataFrame({
                "date": "9/11/2001",
                "responsiveness": 0,
                "message": messages
            })
            
        return featureConstruction(dataFrame= df, createDataFrame=False).get_dataframe()

    def __predict__(self):
        user_messages = []

        for _ in range(5):
            user_messages.append(input("Scrivi un messaggio:\n"))

            df = self.dataframe_for_messages(user_messages)

            predictedUser = self.__trainedModel.predict(df)

            print(predictedUser)

            



if __name__ == "__main__":
    exit()
