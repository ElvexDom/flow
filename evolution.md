Voici une version percutante pour tes élèves, suivie du code pour le décorateur.
### Challenge : Benchmarking Algorithmique

Vous avez utilisé les modèles linéaires de base. Comparez vos résultats (métriques) et vos **temps de calcul** entre les modèles historiques et les méthodes d'ensemble modernes.

* **Régression** : Linéaire vs **Random Forest**, **SVR**, **XGBoost** / **LightGBM**.
* **Classification** : Logistique vs **SVM**, **Random Forest**, **XGBoost** / **CatBoost**.
* **Clustering & Anomalies** : GMM, **DBSCAN** et **Isolation Forest**.

**Objectif :** Loggez tout sur **MLFlow** et déterminez quel algorithme offre le meilleur rapport "Précision / Temps".

---

### Utilisez un Décorateur de tyoe Chronomètre

Ce décorateur, si il marche toujours, calcule le temps de l'entraînement et l'envoie directement sur la console et dans MLFlow si une session est ouverte.

```python
import time
import mlflow
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Déclenchement du chrono
        start_time = time.perf_counter()
        
        # Exécution de la fonction (ex: pipe.fit)
        result = func(*args, **kwargs)
        
        # Fin du chrono
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Affichage écran
        print(f"⏱️ '{func.__name__}' exécuté en {duration:.4f} secondes")
        
        # Log MLFlow (si un run est actif)
        if mlflow.active_run():
            mlflow.log_metric("execution_time_sec", duration)
            
        return result
    return wrapper

# --- EXEMPLE D'UTILISATION ---

Un décorateur se pose sur la définition d'une fonction, pas sur son exécution

@monitor_performance
def train_model(model, X, y):
    return model.fit(X, y)

# puis dans le code :
train_model(pipe, X_train, y_train)

```