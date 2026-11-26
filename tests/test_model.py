"""ETAPA 4 - Pruebas automatizadas del pipeline.

Diez pruebas que se ejecutan DESPUES del entrenamiento. Mientras la validacion
de datos revisa la entrada contra el contrato, estas pruebas revisan la salida:
archivos generados, modelo cargable y criterios de calidad del desempeno.

Umbrales tomados de data_contract.json -> quality_gates.
Ejecucion: pytest -v  (o python run_pipeline.py, etapa 4)
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from config import CONTRACT_FILE, MODEL_FILE, TEST_FILE, TRAIN_FILE

TARGET = "species"
CONTRATO = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
GATES = CONTRATO["quality_gates"]
COLUMNAS_ESPERADAS = list(CONTRATO["feature_order"]) + [TARGET]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    if not TRAIN_FILE.exists():
        pytest.fail("No existe data/train.csv: ejecute la etapa de preparacion")
    return pd.read_csv(TRAIN_FILE)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    if not TEST_FILE.exists():
        pytest.fail("No existe data/test.csv: ejecute la etapa de preparacion")
    return pd.read_csv(TEST_FILE)


@pytest.fixture(scope="module")
def modelo():
    if not MODEL_FILE.exists():
        pytest.fail("No existe models/model.pkl: ejecute la etapa de entrenamiento")
    return joblib.load(MODEL_FILE)


# --------------------------------------------------------------------------- #
# 1-5. Pruebas sobre los datos procesados
# --------------------------------------------------------------------------- #
def test_01_archivos_procesados_existen():
    """Los archivos de entrenamiento y prueba fueron generados."""
    assert TRAIN_FILE.exists(), "falta data/train.csv"
    assert TEST_FILE.exists(), "falta data/test.csv"


def test_02_archivos_no_estan_vacios(train_df, test_df):
    """Ambos archivos contienen filas."""
    assert len(train_df) > 0, "data/train.csv esta vacio"
    assert len(test_df) > 0, "data/test.csv esta vacio"


def test_03_columnas_coinciden_con_el_contrato(train_df, test_df):
    """El esquema de los archivos es el declarado en el contrato de datos."""
    assert list(train_df.columns) == COLUMNAS_ESPERADAS
    assert list(test_df.columns) == COLUMNAS_ESPERADAS


def test_04_sin_valores_nulos(train_df, test_df):
    """No quedan nulos despues de la preparacion."""
    assert train_df.isnull().sum().sum() == 0, "hay nulos en train.csv"
    assert test_df.isnull().sum().sum() == 0, "hay nulos en test.csv"


def test_05_variable_objetivo_con_dominio_valido(train_df, test_df):
    """La variable objetivo existe y solo contiene las clases permitidas."""
    permitidas = set(CONTRATO["columns"][TARGET]["allowed_values"])
    assert TARGET in train_df.columns and TARGET in test_df.columns
    assert set(train_df[TARGET]).issubset(permitidas)
    assert set(test_df[TARGET]).issubset(permitidas)


# --------------------------------------------------------------------------- #
# 6-8. Pruebas sobre el modelo
# --------------------------------------------------------------------------- #
def test_06_archivo_del_modelo_existe():
    """El artefacto del modelo quedo guardado para la inferencia."""
    assert MODEL_FILE.exists(), "falta models/model.pkl"


def test_07_modelo_se_carga_y_predice(modelo, test_df):
    """El modelo se deserializa y produce una prediccion por fila."""
    X = test_df.drop(columns=[TARGET])
    predicciones = modelo.predict(X)
    assert len(predicciones) == len(test_df)


def test_08_accuracy_supera_umbral_minimo(modelo, test_df):
    """El accuracy en prueba cumple el umbral minimo del contrato."""
    X, y = test_df.drop(columns=[TARGET]), test_df[TARGET]
    accuracy = accuracy_score(y, modelo.predict(X))
    minimo = GATES["min_accuracy_test"]
    assert accuracy >= minimo, (
        f"accuracy {accuracy:.4f} inferior al minimo exigido {minimo}")


# --------------------------------------------------------------------------- #
# 9-10. Pruebas de integridad experimental (aporte propio)
# --------------------------------------------------------------------------- #
def test_09_sin_fuga_de_datos_entre_train_y_test(train_df, test_df):
    """Ninguna fila de prueba aparece en entrenamiento (control de fuga de datos).

    Ninguna regla del contrato detecta este problema: un registro de prueba
    copiado en entrenamiento cumple tipos, nulos y rangos, pero infla el
    accuracy porque el modelo recuerda en lugar de predecir.
    """
    filas_train = set(map(tuple, train_df.itertuples(index=False, name=None)))
    filas_test = set(map(tuple, test_df.itertuples(index=False, name=None)))
    interseccion = filas_train & filas_test
    permitido = GATES["train_test_overlap_allowed"]
    assert len(interseccion) <= permitido, (
        f"{len(interseccion)} filas de prueba estan tambien en entrenamiento")


def test_10_brecha_train_test_no_indica_sobreajuste(modelo, train_df, test_df):
    """La diferencia entre accuracy de entrenamiento y de prueba es aceptable.

    Un accuracy de prueba perfecto es una alerta, no un premio: contrastar
    ambas metricas permite detectar sobreajuste de forma automatizada.
    """
    acc_train = accuracy_score(train_df[TARGET], modelo.predict(train_df.drop(columns=[TARGET])))
    acc_test = accuracy_score(test_df[TARGET], modelo.predict(test_df.drop(columns=[TARGET])))
    brecha = abs(acc_train - acc_test)
    maxima = GATES["max_gap_train_test_accuracy"]
    assert brecha <= maxima, (
        f"brecha de {brecha:.4f} entre entrenamiento ({acc_train:.4f}) y "
        f"prueba ({acc_test:.4f}); el maximo admitido es {maxima}")
