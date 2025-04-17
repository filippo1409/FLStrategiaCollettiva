"""et: A Flower / sklearn app."""

import numpy as np
import pickle
import base64
import os
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
import sklearn.utils
import pandas as pd


def load_data(partition_id: int, num_partitions: int):
    try:
        dataset_path = "et/datasetUnito.csv"
        print(f"Caricamento del dataset da {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        
        # Converti colonna marker in valori binari
        df['marker'] = df['marker'].astype(str).str.lower().apply(
            lambda x: 0 if 'natural' in x else 1
        )

        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Sostituisci i valori NaN con la mediana
        for col in df.columns:
            if col != 'marker':
                # Calcola la mediana
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        X = df.drop('marker', axis=1).values
        y = df['marker'].values
        
        # Applica SMOTE per bilanciare le classi
        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Shuffle dei dati per una distribuzione casuale
        X, y = sklearn.utils.shuffle(X_res, y_res, random_state=42)
        
        print(f"Dataset caricato: {X.shape[0]} esempi, {X.shape[1]} feature")
        print(f"Distribuzione delle classi: {np.bincount(y)}")
        
        # Dividi il dataset in partizioni
        total_samples = X.shape[0]
        samples_per_partition = total_samples // num_partitions
        
        start_idx = partition_id * samples_per_partition
        end_idx = start_idx + samples_per_partition if partition_id < num_partitions - 1 else total_samples
        
        X_partition = X[start_idx:end_idx]
        y_partition = y[start_idx:end_idx]
        
        print(f"Partizione {partition_id}: {X_partition.shape[0]} esempi")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_partition, y_partition, test_size=0.2, random_state=21
        )
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        import traceback
        traceback.print_exc()
        # Restituisci array vuoti in caso di errore
        return np.array([]), np.array([]), np.array([]), np.array([])


def get_model(n_estimators: int, random_state: int):
    print(f"Creazione modello ExtraTrees con n_estimators = {n_estimators}, random_state = {random_state}")
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )


def get_model_params(model):
    buffer = BytesIO()
    pickle.dump(model, buffer)
    model_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return [np.array([len(model_str)], dtype=np.int32), np.frombuffer(model_str.encode('utf-8'), dtype=np.uint8)]


def set_model_params(model, params):
    if not params or len(params) < 2:
        return model
    str_len = params[0][0]
    model_str = params[1].tobytes()[:str_len].decode('utf-8')
    
    try:
        buffer = BytesIO(base64.b64decode(model_str))
        loaded_model = pickle.load(buffer)
        model.__dict__.update(loaded_model.__dict__)
        return model
            
    except Exception as e:
        print(f"Errore durante la deserializzazione del modello: {e}")
        # In caso di errore, restituiamo il modello originale
        return model

