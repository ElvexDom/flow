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
        # if mlflow.active_run():
        mlflow.log_metric("execution_time_sec", duration)
            
        return result
    return wrapper