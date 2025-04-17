"""et: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import FitRes, Parameters, EvaluateRes
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import pickle
import base64
import json
from io import BytesIO
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
import copy
from et.task import get_model, get_model_params, set_model_params


class ExtraTreesEnsembleStrategy(FedAvg):
    """Strategia personalizzata per ExtraTrees che utilizza VotingClassifier."""
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregazione dei modelli ExtraTrees usando VotingClassifier."""
        if not results:
            return None, {}
        
        # Calcoliamo il numero totale di esempi
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        print(f"Round {server_round}: {len(results)} client, {total_examples} esempi totali")
        
        # Calcoliamo i pesi per ogni client in base al numero di esempi e alla loro accuratezza
        weights = []
        for _, fit_res in results:
            # Usiamo una combinazione di numero di esempi e accuratezza per i pesi
            base_weight = fit_res.num_examples / total_examples
            # Se abbiamo metriche di accuratezza, le usiamo per pesare ulteriormente
            if hasattr(fit_res, 'metrics') and 'accuracy' in fit_res.metrics:
                accuracy_weight = fit_res.metrics['accuracy']
                # Combiniamo i pesi con un fattore di bilanciamento
                final_weight = 0.7 * base_weight + 0.3 * accuracy_weight
            else:
                final_weight = base_weight
            weights.append(final_weight)
        
        # Normalizziamo i pesi
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        client_models = []
        for i, (_, fit_res) in enumerate(results):
            params = parameters_to_ndarrays(fit_res.parameters)
            
            if len(params) >= 2:
                str_len = params[0][0]
                model_str = params[1].tobytes()[:str_len].decode('utf-8')
                
                try:
                    buffer = BytesIO(base64.b64decode(model_str))
                    client_model = pickle.load(buffer)
                    client_models.append((client_model, weights[i]))
                    print(f"Modello del client {i} caricato con successo, peso: {weights[i]:.4f}")
                except Exception as e:
                    print(f"Errore nel caricare il modello del client {i}: {e}")
        
        if not client_models:
            print("Nessun modello client valido da aggregare")
            return None, {}
        
        print(f"Aggregazione di {len(client_models)} modelli")
        
        # STRATEGIA DI ENSEMBLE CON VOTINGCLASSIFIER
        try:
            if server_round > 1 and len(client_models) >= 2:
                # Creazione di un ensemble pesato di modelli
                print("Applicando strategia di ensemble voting...")
                
                # Creiamo una lista di modelli con i loro pesi
                model_info = []
                for i, (model, weight) in enumerate(client_models):
                    model_info.append({
                        "index": i,
                        "weight": weight,
                        "model": model
                    })
                
                # Prepariamo gli stimatori per il VotingClassifier
                voting_weights = [info["weight"] for info in model_info]
                estimators = [
                    (f"client_{i}", info["model"]) 
                    for i, info in enumerate(model_info)
                ]
                
                # Creiamo un modello VotingClassifier che combina i modelli client
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=voting_weights
                )
                
                # Creiamo una copia del modello di base
                base_model = copy.deepcopy(model_info[0]["model"])
                
                # Memorizziamo l'ensemble come attributo e aggiungiamo informazioni sui modelli aggregati
                base_model.voting_ensemble_ = ensemble
                base_model.ensemble_info_ = {
                    "n_models": len(estimators),
                    "weights": voting_weights,
                    "server_round": server_round
                }
                
                print(f"Modello ensemble creato con {len(estimators)} modelli client")
                aggregated_model = base_model
            else:
                # Nel primo round o se abbiamo meno di 2 client, usiamo il modello con più peso
                best_model, best_weight = max(client_models, key=lambda x: x[1])
                aggregated_model = copy.deepcopy(best_model)
                print(f"Utilizzato il modello client con peso maggiore: {best_weight:.4f}")
        except Exception as e:
            print(f"Errore durante la creazione dell'ensemble: {e}")
            # Fallback: utilizziamo il modello con più dati
            try:
                best_model, _ = max(client_models, key=lambda x: x[1])
                aggregated_model = copy.deepcopy(best_model)
            except:
                # Se anche questo fallisce, creiamo un modello nuovo
                aggregated_model = get_model(n_estimators=200, random_state=42)  # Aumentato il numero di alberi di default
            print("Fallback: utilizzato il modello del client con più dati o un nuovo modello")
        
        # Serializziamo il modello aggregato
        return ndarrays_to_parameters(get_model_params(aggregated_model)), {}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggrega i risultati di valutazione dai client."""
        if not results:
            return None, {}
        
        # Aggregazione standard delle metriche di valutazione
        loss_aggregated = weighted_average([
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ])
        
        # Aggreghiamo le metriche di accuratezza
        metrics_aggregated = {"accuracy": 0.0}  # valore di default
        for _, evaluate_res in results:
            for key, value in evaluate_res.metrics.items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = []
                if isinstance(metrics_aggregated[key], list):
                    metrics_aggregated[key].append((evaluate_res.num_examples, value))
                else:
                    metrics_aggregated[key] = [(evaluate_res.num_examples, value)]
        
        # Calcoliamo la media pesata per ogni metrica
        for key, values in metrics_aggregated.items():
            if isinstance(values, list):
                metrics_aggregated[key] = weighted_average(values)
        
        print(f"Round {server_round}: Loss aggregata = {loss_aggregated}, Metriche = {metrics_aggregated}")
        
        return loss_aggregated, metrics_aggregated


def weighted_average(metrics: List[Tuple[int, float]]) -> float:
    """Calcola la media pesata delle metriche."""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    return sum([num_examples * metric for num_examples, metric in metrics]) / total_examples


def evaluate_metrics_aggregation(eval_metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggrega le metriche di valutazione dai client."""
    # Otteniamo il numero totale di esempi
    total_examples = sum([num_examples for num_examples, _ in eval_metrics])
    
    # Aggreghiamo le metriche pesate per il numero di esempi
    agg_metrics = {}
    for metric_name in eval_metrics[0][1].keys():
        weighted_sum = sum([
            num_examples * metrics[metric_name]
            for num_examples, metrics in eval_metrics
        ])
        agg_metrics[metric_name] = weighted_sum / total_examples
    
    return agg_metrics


def config_func(server_round: int) -> Dict[str, float]:
    """Restituisce la configurazione per training/evaluation."""
    return {"server_round": server_round}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Create ExtraTrees Model
    n_estimators = context.run_config["n-estimators"]
    random_state = context.run_config["random-state"]
    model = get_model(n_estimators, random_state)

    # Otteniamo i parametri iniziali del modello
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Utilizziamo la nostra strategia personalizzata di ensemble
    strategy = ExtraTreesEnsembleStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=4,
        min_fit_clients=4,
        min_evaluate_clients=4,
        initial_parameters=initial_parameters,
        accept_failures=True,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)