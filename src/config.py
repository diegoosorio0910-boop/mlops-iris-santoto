"""Configuracion central del proyecto MLOps-Iris-Santoto.

Un unico lugar define rutas, semilla y nombres de experimento. Cualquier otro
modulo importa desde aqui, de modo que no existan valores duplicados ni
"numeros magicos" repartidos por el codigo (principio de trazabilidad,
seccion 1.7 del libro del espacio academico).
"""
from pathlib import Path

# --- Rutas base -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

RAW_FILE = RAW_DIR / "iris.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
BACKUP_DIR = DATA_DIR / "_backup"

CONTRACT_FILE = ROOT_DIR / "data_contract.json"
MODEL_FILE = MODELS_DIR / "model.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"

# --- Reproducibilidad -------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Hiperparametros del baseline ------------------------------------------
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "random_state": RANDOM_STATE,
}

# --- MLflow -----------------------------------------------------------------
MLFLOW_EXPERIMENT = "iris-pipeline-semana2"
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"


def asegurar_directorios() -> None:
    """Crea las carpetas de salida si no existen (ejecucion desde cero)."""
    for carpeta in (DATA_DIR, RAW_DIR, MODELS_DIR, LOGS_DIR):
        carpeta.mkdir(parents=True, exist_ok=True)
