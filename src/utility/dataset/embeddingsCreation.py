import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader

class EmbeddingsCreation:
    def __init__(self, dataFrame: pd.DataFrame, datasetPath: str, saveDataFrame :bool = True):
        self.DATASET_PATH = self.__check_dataset_path(datasetPath)
        self.__dataFrame = dataFrame

        self.__create_embeddings()

        if saveDataFrame:
            self .__save_dataframe()


    def get_dataframe(self):
        return self.__dataFrame
    
    @staticmethod
    def __check_dataset_path(dataset_path: str) -> str:
        """Controlla se il percorso del dataset esiste"""

        dir_name = os.path.dirname(dataset_path)
        base_name = os.path.basename(dataset_path)

        prefix = "embeddings_"

        if base_name.startswith(prefix):
            new_name = base_name
        else:
            new_name = prefix + base_name

        actual_path = os.path.join(dir_name, new_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        return actual_path

    def __create_embeddings(self):
        """Crea gli embeddings per ogni messaggio nel dataframe."""
        
        # Rimuovo la colonna "responsiveness" 
        self.__dataFrame = self.__dataFrame.drop(columns=["responsiveness"], errors="ignore")

        # Inizializza il tokenizer e il modello BERT
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name)
        bert_model.eval()

        # Tokenizza i messaggi
        # Processa i messaggi in batch
        batch_size = 32  
        messages = self.__dataFrame["message"].tolist()
        data_loader = DataLoader(messages, batch_size=batch_size, shuffle=False)

        all_cls_embeds = []

        for batch in data_loader:
            encodings = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = bert_model(**encodings)
            
            # Extract CLS token embeddings
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_cls_embeds.append(cls_embeds)

        # Concatenate all embeddings
        all_cls_embeds = np.vstack(all_cls_embeds)


        n_dims = all_cls_embeds.shape[1]
        # n_dims = 768 for BERT base model

        # Rinomina le colonne embed_0, embed_1, …, embed_767
        embed_cols = [f"embed_{i}" for i in range(n_dims)]
        df_emb_vals = pd.DataFrame(all_cls_embeds, columns=embed_cols)

        # Concatena 
        self.__dataFrame = pd.concat([self.__dataFrame[["user"]].reset_index(drop=True),
                                        df_emb_vals.reset_index(drop=True)],
                                    axis=1)
        
    def __save_dataframe(self):
        """Salva il dataframe in formato parquet."""
        self.__dataFrame.to_parquet(self.DATASET_PATH, index=False)