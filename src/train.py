"""ETAPA 3 - Entrenamiento y registro del experimento.

Responsabilidad unica: entrenar el baseline (Random Forest) con los datos ya
validados, registrar parametros, metricas y el artefacto en MLflow, y guardar
models/model.pkl para la futura inferencia.

Salidas: models/model.pkl, models/metrics.json, run registrado en MLflow
Codigos de salida: 0 = exito, 1 = error controlado
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from config import (
    METRICS_FILE,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_FILE,
    MODEL_PARAMS,
    RANDOM_STATE,
    TEST_FILE,
    TRAIN_FILE,
    asegurar_directorios,
)

TARGET = "species"


def cargar(ruta):
    df = pd.read_csv(ruta)
    return df.drop(columns=[TARGET]), df[TARGET]


def registrar_modelo(mlflow, modelo) -> None:
    """Registra el artefacto del modelo en MLflow de forma compatible entre versiones.

    MLflow cambio el nombre del parametro de ruta (artifact_path -> name) y, desde
    la version 3, exige declarar como confiables los tipos de arbol de scikit-learn.
    Se intentan las variantes en orden y se usa la primera que funcione, de modo que
    el pipeline sea reproducible en distintos entornos.
    """
    import mlflow.sklearn

    intentos = [
        {"name": "model", "skops_trusted_types": ["sklearn.tree._tree.Tree"]},
        {"name": "model", "serialization_format": "cloudpickle"},
        {"artifact_path": "model"},
    ]
    for kwargs in intentos:
        try:
            mlflow.sklearn.log_model(modelo, **kwargs)
            return
        except TypeError:
            continue
        except Exception:  # noqa: BLE001 - se prueba la siguiente variante
            continue
    # Ultimo recurso: se adjunta el .pkl como artefacto plano
    mlflow.log_artifact(str(MODEL_FILE), artifact_path="model")


def main() -> int:
    asegurar_directorios()

    X_train, y_train = cargar(TRAIN_FILE)
    X_test, y_test = cargar(TEST_FILE)
    print(f"  Entrenamiento: {X_train.shape[0]} filas | Prueba: {X_test.shape[0]} filas")

    modelo = RandomForestClassifier(**MODEL_PARAMS)
    modelo.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_test = accuracy_score(y_test, modelo.predict(X_test))
    f1_test = f1_score(y_test, modelo.predict(X_test), average="macro")
    cv = cross_val_score(modelo, X_train, y_train, cv=5)

    metricas = {
        "accuracy_train": round(float(acc_train), 4),
        "accuracy_test": round(float(acc_test), 4),
        "f1_macro_test": round(float(f1_test), 4),
        "cv_accuracy_mean": round(float(cv.mean()), 4),
        "cv_accuracy_std": round(float(cv.std()), 4),
        "gap_train_test": round(float(acc_train - acc_test), 4),
    }

    print(f"  Accuracy entrenamiento: {metricas['accuracy_train']:.4f}")
    print(f"  Accuracy prueba:        {metricas['accuracy_test']:.4f}")
    print(f"  F1 macro prueba:        {metricas['f1_macro_test']:.4f}")
    print(f"  Validacion cruzada 5-fold: {metricas['cv_accuracy_mean']:.4f} "
          f"(+/- {metricas['cv_accuracy_std']:.4f})")
    print(f"  Brecha entrenamiento-prueba: {metricas['gap_train_test']:.4f}")

    # --- Artefactos locales -------------------------------------------------
    joblib.dump(modelo, MODEL_FILE)
    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": "RandomForestClassifier",
        "parametros": MODEL_PARAMS,
        "metricas": metricas,
        "entorno": {
            "python": platform.python_version(),
            "sistema": f"{platform.system()} {platform.release()}",
        },
        "semilla": RANDOM_STATE,
    }
    METRICS_FILE.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Modelo guardado en: models/{MODEL_FILE.name}")

    # --- Registro en MLflow (trazabilidad del experimento) ------------------
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"baseline_rf_{datetime.now():%Y%m%d_%H%M%S}"):
            mlflow.log_params(MODEL_PARAMS)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("contrato_version", "1.1.0")
            mlflow.log_metrics(metricas)
            registrar_modelo(mlflow, modelo)
            mlflow.log_artifact(str(METRICS_FILE))
        print(f"  Run registrado en MLflow | experimento: {MLFLOW_EXPERIMENT}")
    except ImportError:
        print("  AVISO: MLflow no esta instalado; el run no se registro.")
        print("         Instale las dependencias con: pip install -r requirements.txt")
    except Exception as error:  # noqa: BLE001
        print(f"  AVISO: no fue posible registrar en MLflow ({error}).")
        print("         El modelo y las metricas locales si quedaron guardados.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as error:
        print(f"  ERROR: falta un archivo de entrada ({error.filename}). "
              "Ejecute primero las etapas de preparacion y validacion.", file=sys.stderr)
        sys.exit(1)
    except Exception as error:  # noqa: BLE001
        print(f"  ERROR en el entrenamiento: {error}", file=sys.stderr)
        sys.exit(1)
