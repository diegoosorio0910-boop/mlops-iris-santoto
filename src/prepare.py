"""ETAPA 1 - Preparacion de datos.

Responsabilidad unica: cargar el dataset Iris desde scikit-learn, renombrar las
columnas a los nombres declarados en el contrato de datos y dividirlo en
entrenamiento y prueba (80/20, estratificado, semilla fija).

Salidas: data/raw/iris.csv, data/train.csv, data/test.csv
Codigos de salida: 0 = exito, 1 = error controlado
"""
from __future__ import annotations

import json
import sys

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from config import (
    CONTRACT_FILE,
    RANDOM_STATE,
    RAW_FILE,
    TEST_FILE,
    TEST_SIZE,
    TRAIN_FILE,
    asegurar_directorios,
)

COLUMNAS = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
}


def cargar_iris() -> pd.DataFrame:
    """Carga Iris desde scikit-learn y aplica los nombres del contrato."""
    bunch = load_iris(as_frame=True)
    df = bunch.frame.rename(columns=COLUMNAS)
    df = df.rename(columns={"target": "species"})
    return df


def orden_contrato(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena las columnas segun feature_order + target del contrato."""
    contrato = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    orden = list(contrato["feature_order"]) + [contrato["target"]]
    return df[orden]


def main() -> int:
    asegurar_directorios()

    df = orden_contrato(cargar_iris())
    df.to_csv(RAW_FILE, index=False)
    print(f"  Dataset original cargado: {len(df)} filas, {len(df.columns)} columnas")
    print(f"  Guardado en: {RAW_FILE.relative_to(RAW_FILE.parents[2])}")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["species"],
    )

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(
        f"  Division estratificada {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}"
        f" con random_state={RANDOM_STATE}"
    )
    print(f"  train.csv: {len(train_df)} filas  |  test.csv: {len(test_df)} filas")
    print("  Distribucion por clase en entrenamiento: "
          + ", ".join(f"{k}={v}" for k, v in sorted(train_df['species'].value_counts().items())))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - se reporta y se corta el pipeline
        print(f"  ERROR en la preparacion de datos: {error}", file=sys.stderr)
        sys.exit(1)
