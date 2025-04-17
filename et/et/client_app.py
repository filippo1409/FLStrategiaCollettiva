"""et: A Flower / sklearn app."""

import warnings
import numpy as np
from sklearn.metrics import log_loss

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from et.task import (
    get_model,
    get_model_params,
    load_data,
    set_model_params,
)


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is_fitted = False

    def fit(self, parameters, config):
        # Impostiamo i parametri del modello
        model = set_model_params(self.model, parameters)
        self.model = model  # Aggiorniamo il riferimento al modello
        
        # Addestriamo il modello ExtraTrees
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
            self.is_fitted = True  # Impostiamo il flag a True dopo l'addestramento
            print(f"Modello addestrato con successo. Numero di esempi: {len(self.X_train)}")

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        # Impostiamo i parametri del modello
        model = set_model_params(self.model, parameters)
        self.model = model  # Aggiorniamo il riferimento al modello
        
        try:
            # Calcoliamo le probabilità per il calcolo della loss
            y_pred_proba = self.model.predict_proba(self.X_test)
            loss = log_loss(self.y_test, y_pred_proba)
            accuracy = self.model.score(self.X_test, self.y_test)
                
            print(f"Valutazione completata. Loss: {loss}, Accuracy: {accuracy}")
            return loss, len(self.X_test), {"accuracy": accuracy}
        except Exception as e:
            print(f"Errore durante la valutazione: {e}")
            # In caso di errore, restituiamo valori di default
            return float('inf'), len(self.X_test), {"accuracy": 0.0}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create ExtraTrees Model
    n_estimators = context.run_config["n-estimators"]
    random_state = context.run_config["random-state"]
    model = get_model(n_estimators, random_state)

    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)