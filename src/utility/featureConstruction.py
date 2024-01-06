import spacy
import emojis
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter, defaultdict


class featureConstruction():

    def __init__(self, dataFrame : pd.DataFrame, datasetPath : str, fileName : str):
        self.DATASET_PATH = datasetPath
        self.FILE_NAME = fileName

        self.__dataFrame = dataFrame
        self.__feature_construction()

    def __write_dataFrame(self):
        # counter = 0

        #fileName = self.FILE_NAME + self.FILE_EXT

        fileName = self.FILE_NAME

        # while os.path.isfile(self.DATASET_PATH + filename):
        #     counter += 1
        #     filename = f"{self.FILE_NAME}{counter}{self.FILE_EXT}"
        
        self.__dataFrame.to_parquet(self.DATASET_PATH + fileName)
        


    def __feature_construction(self):
        '''Crea nuove feature: un insieme di metadati relativo ad ogni messaggio.'''
        # Inizializza analizzatore NLP
        # Bisogna prima scaricare i moduli (sm -> small, md -> medium, lg -> large): 
        #   python3 -m spacy download en_core_web_sm
        #   python3 -m spacy download it_core_news_lg
        # ATTENZIONE: Il vocabolario inglese contiene anche parole italiane e viceversa
        nlp_it = spacy.load("it_core_news_lg")
        nlp_en = spacy.load("en_core_web_sm")

        # Lista di vocaboli italiani e inglese in minuscolo
        italian_words = {w.lower() for w in nlp_it.vocab.strings}
        english_words = {w.lower() for w in nlp_en.vocab.strings}

        # Lista tag POS (Part-of-speech)
        POS_LIST = [
            "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", 
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", 
            "SYM", "VERB", "X", "SPACE"
        ]

        # Inizializza sentiment/emotion classifier
        # sentiment_classifier = SentimentClassifier()
        # sentiment_mapping = {'negative':0, 'positive':1}
        # emotion_classifier = EmotionClassifier()
        # emotion_mapping = {"anger":0, "fear":1 , "joy":2, "sadness":3 }


        def uppercase_count(m):
            """Conta il numero di caratteri in maiuscolo in un messaggio."""
            return sum(1 for c in m if c.isupper())

        def char_count(m): 
            """Conta il numero di caratteri in un message."""
            return len(m)

        def words_length(m):
            """Conta la lunghezza media delle parole in un messaggio."""
            parole = m.split()
            
            if not parole:
                return 0  # Restituisce 0 se la stringa è vuota
            
            lunghezza_totale = sum(len(parola) for parola in parole)
            lunghezza_media = lunghezza_totale / len(parole)
            
            return round(lunghezza_media, 2)

        def emojis_count(s):
            """Conta il numero di emojii in un messaggio."""
            return emojis.count(s)

        def unique_emojis_count(s):
            """Conta il numero di emojii uniche in un messaggio."""
            return emojis.count(s, unique=True)

        def vocabulary_count(nlp_message, vocabulary):
            '''
                Indica quanto una persona parla italiano "pulito" 
                    oppure quanti inglesismi usa (in base al vocabolario in input)
                
                cioè il rapporto fra parole presenti nel vocabolario 
                    e il numero totale di parole in un messaggio.
            '''
            count = 0
            total_count = 0

            for token in nlp_message:
                if token.is_alpha: # per filtrare numeri e simboli
                    total_count += 1

                    if token.text.lower() in vocabulary:
                        count += 1

            # Frase senza parole
            if total_count == 0:
                return 0

            return round(count/total_count, 2)

        def message_composition(nlp_message):
            """
                Calcola la percentuale di tag POS (Part-of-speech) presenti in un messaggio.
                Es. "Mario mangia la pasta" 
                    conta il numero di pronomi, verbi, articoli, sostantivi... -> {'PROPN': 1, 'VERB': 1, 'DET': 1, 'NOUN': 1}
                    calcola la percentuale di ognuno -> {'VERB': 0.2, 'SCONJ': 0.2, 'PROPN': 0.2, 'PUNCT': 0.2, 'DET': 0.2, 'NOUN': 0.2}
                
                Lista di tag POS:
                ADJ: Adjective, e.g., big, old, green, incomprehensible, first.
                ADP: Adposition, e.g., in, to, during.
                ADV: Adverb, e.g., very, tomorrow, down, where, there1.
                AUX: Auxiliary verb, e.g., is, has (done), will (do), should (do)1.
                CONJ: Conjunction.
                CCONJ: Coordinating conjunction, e.g., and, or, but.
                DET: Determiner, e.g., a, an, the.
                INTJ: Interjection, e.g., psst, ouch, bravo, hello.
                NOUN: Noun, e.g., girl, cat, tree, air, beauty.
                NUM: Numeral, e.g., 1, 2017, one, seventy-seven, IV, MMXIV1.
                PART: Particle.
                PRON: Pronoun, e.g., I, you, he, she, myself, themselves, somebody1.
                PROPN: Proper noun, e.g., Mary, John, London, NATO, HBO1.
                PUNCT: Punctuation, e.g., ., (, ), ?.
                SCONJ: Subordinating conjunction, e.g., if, while, that.
                SYM: Symbol, e.g., *$, %, §, ©, +, −, ×, ÷, =, :).
                VERB: Verb, e.g., run, runs, running, eat, ate, eating.
                X: Other.
                SPACE: Space.
            """
            
            # Conta token per tipo
            c = Counter(([token.pos_ for token in nlp_message])) 

            # Numero totale di token
            sbase = sum(c.values())

            # Calcola percentuale di quel tipo di token nel messaggio
            d = defaultdict(int) # dizionario con valori di default 0
            for el, cnt in c.items():
                d[el] = round(cnt/sbase, 2)

            return d
        
        def first_word_type(nlp_message):
            """Restituisce l'id del tag POS della prima parola di un messaggio."""
            if (nlp_message.text == ""):
                return -1

            return POS_LIST.index(nlp_message[0].pos_)
        
        def sentiment(m):
            """Restituisce l'id del sentiment del messaggio descritto in sentiment_mapping."""
            return sentiment_mapping[sentiment_classifier.predict([m])[0]]
        
        def emotion(m):
            """Restituisce l'id dell'emotion del messaggio descritto in emotion_mapping."""
            return emotion_mapping[emotion_classifier.predict([m])[0]]

        # Calcola nuove feature
        composition_list = []
        uppercase_list = []
        length_list = []
        word_length_list = []
        emojii_list = []
        unique_emojii_list = []
        italianness_list = []
        englishness_list = []
        first_word_type_list = []
        # emotion_list = []
        # sentiment_list = []

        for message in tqdm(self.__dataFrame['message']):
            uppercase_list.append(uppercase_count(message))
            length_list.append(char_count(message))
            word_length_list.append(words_length(message))
            emojii_list.append(emojis_count(message))
            unique_emojii_list.append(unique_emojis_count(message))
            # emotion_list.append(emotion(message))
            # sentiment_list.append(sentiment(message))

            nlp_it_message = nlp_it(message)
            nlp_en_message = nlp_en(message)

            composition_list.append(message_composition(nlp_it_message))
            first_word_type_list.append(first_word_type(nlp_it_message))
            italianness_list.append(vocabulary_count(nlp_it_message, italian_words))
            englishness_list.append(vocabulary_count(nlp_en_message, english_words))

        # Aggiungi nuove colonne al dataframe
        self.__dataFrame["uppercase"] = uppercase_list
        self.__dataFrame["length"] = length_list
        self.__dataFrame["word_length"] = word_length_list
        self.__dataFrame["emojii"] = emojii_list
        self.__dataFrame["unique_emojii"] = unique_emojii_list
        self.__dataFrame["italianness"] = italianness_list
        self.__dataFrame["englishness"] = englishness_list
        # df["first_word_type"] = first_word_type_list
        # df["sentiment"] = sentiment_list
        # df["emotion"] = emotion_list

        for pos in POS_LIST:
            self.__dataFrame[pos] = [d[pos] for d in composition_list]

        self.__write_dataFrame()