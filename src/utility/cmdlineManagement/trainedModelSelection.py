import os
import logging
import pandas as pd
import inquirer
import torch
from utility.exceptions import ExtensionError
from utility.model.modelTraining_embeddings import CNN1D
from utility.model.modelTraining_feature import ModelTraining
from joblib import load


class TrainedModelSelection:
    MODEL_PATH = "../models/"

    @classmethod
    def __show_models(cls):
        # Seleziona solo i file
        all_file = [file for file in os.listdir(cls.MODEL_PATH) if os.path.isfile(os.path.join(cls.MODEL_PATH, file))]

        # Seleziona solo i modelli
        models = [model for model in all_file if model.endswith(".joblib") or model.endswith(".pth")]

        # Ordina i modelli in base alla data di creazione
        models = sorted(models, key=lambda x: os.path.getctime(os.path.join(cls.MODEL_PATH, x)), reverse=True)
        
        return models

    @classmethod
    def __load_model(cls, index, models):
        if 0 <= index < len(models):
            selected_model = models[index]

            if not selected_model.endswith(".joblib"):
                raise ExtensionError("Il modello deve essere in formato .joblib")

            path = f"{cls.MODEL_PATH}{selected_model}"

            # Carica il model
            return selected_model, load(path)
        else:
            raise ValueError("ID del modello non valido.")
        
    @classmethod
    def __load_torch_model(cls, index, models):
        if 0 <= index < len(models):
            selected_model = models[index]
            path = f"{cls.MODEL_PATH}{selected_model}"

            checkpoint = torch.load(path, weights_only=False)
            class_names = checkpoint.get('class_names', None)
            num_classes = len(class_names)

            model = CNN1D(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            return model

        else:
            raise ValueError("ID del modello non valido.")

    @classmethod
    def select_model(cls):
        models = cls.__show_models()

        model_selection = [
            inquirer.List('model',
                message="Seleziona il modello da usare",
                choices=models
            ),
        ]

        model = inquirer.prompt(model_selection)

        # Ottieni model
        model_name = model["model"]
        menu_entry_index = models.index(model_name)

        print(f"Modello selezionato: {model_name}")
        # LOGGING:: Stampa il modello (e lo scaler) selezionato
        logging.info(f"Modello usato: {model_name}")

        if model_name.endswith(".joblib"):
            return cls.__load_model(menu_entry_index, models)
        elif model_name.endswith(".pth"):
            return cls.__load_torch_model(menu_entry_index, models)
        

    @property
    def model(self):
        return self.__model