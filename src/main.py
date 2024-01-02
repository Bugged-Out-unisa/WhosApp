import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import spacy
import emojis
from utility.extractChat import ExtractChat
from utility.dataFrameProcess import DataFrameProcessor
from feel_it import EmotionClassifier, SentimentClassifier


# Path della cartella delle chat
# dove verranno analizzati in automatico tutti i file al suo interno
DATA_PATH = "../rawdata"


def read_all_files():
    '''Restituisce una stringa contenente il contenuto di tutti i file nella cartella DATA_PATH.'''
    rawdata = ""
    
    # Ottieni lista dei file nella cartella
    files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]

    # Concatena contenuto di ogni file nella cartella
    for file_name in tqdm(files):
        f = open(os.path.join(DATA_PATH, file_name), 'r', encoding='utf-8')
        rawdata += f.read()
        f.close()

    return rawdata

def feature_construction(df):
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
    sentiment_classifier = SentimentClassifier()
    sentiment_mapping = {'negative':0, 'positive':1}
    emotion_classifier = EmotionClassifier()
    emotion_mapping = {"anger":0, "fear":1 , "joy":2, "sadness":3 }


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

    for message in tqdm(df['message']):
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
    df["uppercase"] = uppercase_list
    df["length"] = length_list
    df["word_length"] = word_length_list
    df["emojii"] = emojii_list
    df["unique_emojii"] = unique_emojii_list
    df["italianness"] = italianness_list
    df["englishness"] = englishness_list
    # df["first_word_type"] = first_word_type_list
    # df["sentiment"] = sentiment_list
    # df["emotion"] = emotion_list

    for pos in POS_LIST:
        df[pos] = [d[pos] for d in composition_list]

def random_forest(df):
    '''Applica random forest sul dataframe.'''
    # Definisci le features (X) e il target (Y) cioè la variabile da prevedere
    X = df.drop(['user', 'date', 'message'], axis=1) # tutto tranne le colonne listate
    y = df["user"]

    # TRASFORMA IL MESSAGGIO IN UNA MATRICE DI FREQUENZA DELLE PAROLE (bag of words)
    # così il modello capisce le parole più utilizzate da un utente
    # ---------------------------------
    # Vettorizza le parole presenti nel messaggio
    vec = CountVectorizer()
    X_message = vec.fit_transform(df['message'])

    # Unisci la matrice al dataframe
    df_words_count = pd.DataFrame(X_message.toarray(), columns=vec.get_feature_names_out())
    X = pd.concat([X, df_words_count], axis=1)
    # ---------------------------------

    # TRAINING CON CROSS VALIDATION
    cv  = 5 # numero di fold (di solito 5 o 10)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

    # Stampa un report sulle metriche di valutazione del modello
    print(f"[INFO] Media delle metriche di valutazione dopo {cv}-fold cross validation:")
    indexes = list(scores.keys())

    for index in indexes:
        print(f"{index}: %0.2f (+/- %0.2f)" % (scores[index].mean(), scores[index].std() * 2))

    # TRAINING CON SPLIT CLASSICO
    test_size = 0.2 # percentuale del dataset di test dopo lo split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Genera un report per il modello addestrato
    print(f"\n[INFO] Report con {int((1-test_size)*100)}% training set e {int(test_size*100)}% test set:")
    
    # Calcola l'accuratezza del modello
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {round(accuracy, 2)}\n')

    # Stampa il classification report
    report = classification_report(y_test, predictions)
    print(report)

    # Stampa le feature più predittive
    n = 20 # numero di feature
    print("\n[INFO] Top {} feature più predittive:".format(n))

    feature_names = X.columns.tolist() # Estrai i nomi di tutte le feature
    importances = model.feature_importances_
    important_features = np.argsort(importances)[::-1]
    top_n_features = important_features[:n]

    for i in top_n_features:
        print(f"{feature_names[i]}: %0.5f" % importances[i])


if __name__ == "__main__":
    print("\n[LOADING] Leggendo le chat dai file grezzi...")
    rawdata = read_all_files()

    print("\n[LOADING] Estraendo informazioni dai dati grezzi...")
    dates, users, messages = ExtractChat(rawdata).extract()

    print("\n[LOADING] Creando il dataframe e applicando data cleaning e undersampling...")
    df = DataFrameProcessor(dates, users, messages).get_dataframe()

    print("\n[LOADING] Applicando feature construction...")
    feature_construction(df)

    print(df.head(25)[['user', 'message', 'italianness', 'englishness']])

    print("\n[LOADING] Addestrando il modello...")
    random_forest(df)