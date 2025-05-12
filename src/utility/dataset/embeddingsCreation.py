import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader

class EmbeddingsCreation:
    def __init__(self, dataFrame: pd.DataFrame, datasetPath: str, saveDataFrame :bool = True, embeddings_strategy = 'mixed'):
        self.DATASET_PATH = self.__check_dataset_path(datasetPath)
        self.__dataFrame = dataFrame

        embeddings_strategy_allowed = ["mixed", "cls", "mean"]

        if embeddings_strategy not in embeddings_strategy_allowed:
            raise ValueError("embeddings_strategy can only have values <", embeddings_strategy_allowed, "> got: ", embeddings_strategy)

        self.embeddings_strategy = embeddings_strategy

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Rimuovo la colonna "responsiveness" 
        self.__dataFrame = self.__dataFrame.drop(columns=["responsiveness"], errors="ignore")

        # Inizializza il tokenizer e il modello BERT
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name)
        bert_model.to(device)

        bert_model.eval()

        # Tokenizza i messaggi
        # Processa i messaggi in batch
        batch_size = 32  
        messages = self.__dataFrame["message"].tolist()
        data_loader = DataLoader(messages, batch_size=batch_size, shuffle=False)

        all_cls_embeds = []
        all_mean_embeds = []

        for batch in tqdm(data_loader):
            encodings = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            if torch.cuda.is_available():
                encodings = {k: v.to(device) for k,v in encodings.items()}

            # Get attention mask for proper averaging
            attention_mask = encodings['attention_mask']

            with torch.no_grad():
                outputs = bert_model(**encodings, output_hidden_states=True)
            
            # Extract CLS token embeddings
            batch_cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_cls_embeds.append(batch_cls_embeds)

            # Mean token embeddings
            # mask padding tokens before averaging

            last_hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            batch_mean_embeds = (sum_embeddings / sum_mask).cpu().numpy()
            all_mean_embeds.append(batch_mean_embeds)

        # Concatenate all embeddings
        all_cls_embeds = np.vstack(all_cls_embeds)
        all_mean_embeds = np.vstack(all_mean_embeds)

        # Choose embedding strategy
        if self.embeddings_strategy == "cls":
            final_embeds = all_cls_embeds
            prefix = "cls_"
        elif self.embeddings_strategy == "mean":
            final_embeds = all_mean_embeds
            prefix = "mean_"
        elif self.embeddings_strategy == "mixed":
            # Concatenate both types of embeddings
            final_embeds = np.hstack([all_cls_embeds, all_mean_embeds])
            prefix = "mixed_"
        else:
            raise ValueError(f"Unknown embedding strategy: {self.embeddings_strategy}")

        # n_dims = 768 for BERT base model
        n_dims = final_embeds.shape[1]
        

        # Rinomina le colonne embed_0, embed_1, …, embed_767
        embed_cols = [f"{prefix}embed_{i}" for i in range(n_dims)]
        df_emb_vals = pd.DataFrame(final_embeds, columns=embed_cols)

        # Concatena 
        self.__dataFrame = pd.concat([self.__dataFrame[["user"]].reset_index(drop=True),
                                        df_emb_vals.reset_index(drop=True)],
                                    axis=1)
        
    def __save_dataframe(self):
        """Salva il dataframe in formato parquet."""
        self.__dataFrame.to_parquet(self.DATASET_PATH, index=False)