#!/usr/bin/env python3
"""Herramienta academica de sabotaje controlado del dataset.

Permite demostrar que el control de calidad funciona: inyecta un error tipico
en data/train.csv y luego se ejecuta el pipeline para evidenciar en que etapa
se detiene. Antes de modificar nada guarda una copia en data/_backup/.

    python romper_datos.py --listar       # muestra los escenarios
    python romper_datos.py 1              # aplica el escenario 1
    python romper_datos.py --restaurar    # devuelve los datos originales

Escenarios 1-4 incumplen el contrato -> los detiene la ETAPA 2 (validacion).
Escenarios 5-6 cumplen el contrato    -> los detiene la ETAPA 4 (pruebas).
Escenario 7 cumple todo               -> no lo detiene nadie (fuga de datos);
                                          lo detecta la prueba 09 de este proyecto.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATA = RAIZ / "data"
TRAIN = DATA / "train.csv"
TEST = DATA / "test.csv"
BACKUP = DATA / "_backup"

ESCENARIOS = {
    1: "Texto en vez de numero: alguien escribio 'cinco' en sepal_length",
    2: "Valor nulo: un sensor dejo vacia la celda petal_width",
    3: "Fuera de rango: un registro quedo en milimetros y no en centimetros",
    4: "Clase inexistente: aparece species = 3",
    5: "Clase ausente: se eliminaron todas las flores virginica del entrenamiento",
    6: "Etiquetas erradas: el 40 % de species quedo mal asignado",
    7: "Fuga de datos: las filas de prueba se copiaron en entrenamiento",
}


def respaldar() -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    for origen in (TRAIN, TEST):
        destino = BACKUP / origen.name
        if origen.exists() and not destino.exists():
            shutil.copy2(origen, destino)


def restaurar() -> int:
    if not BACKUP.exists():
        print("No hay respaldo: ejecute 'python run_pipeline.py' para regenerar los datos.")
        return 1
    for archivo in BACKUP.glob("*.csv"):
        shutil.copy2(archivo, DATA / archivo.name)
    print("Datos originales restaurados desde data/_backup/")
    return 0


def leer(ruta: Path):
    with ruta.open(newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        return lector.fieldnames, list(lector)


def escribir(ruta: Path, campos, filas) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def aplicar(numero: int) -> int:
    if numero not in ESCENARIOS:
        print(f"Escenario {numero} no existe. Use --listar.")
        return 1
    if not TRAIN.exists():
        print("No existen los datos. Ejecute primero: python run_pipeline.py")
        return 1

    respaldar()
    campos, filas = leer(TRAIN)
    random.seed(42)

    if numero == 1:
        filas[1]["sepal_length"] = "cinco"
    elif numero == 2:
        filas[3]["petal_width"] = ""
    elif numero == 3:
        for columna in ("sepal_length", "sepal_width", "petal_length", "petal_width"):
            filas[4][columna] = str(round(float(filas[4][columna]) * 10, 1))
    elif numero == 4:
        filas[5]["species"] = "3"
    elif numero == 5:
        filas = [f for f in filas if f["species"] != "2"]
    elif numero == 6:
        indices = random.sample(range(len(filas)), int(len(filas) * 0.4))
        for i in indices:
            actual = int(filas[i]["species"])
            filas[i]["species"] = str((actual + 1) % 3)
    elif numero == 7:
        _, filas_test = leer(TEST)
        filas = filas + filas_test

    escribir(TRAIN, campos, filas)
    print(f"Escenario {numero} aplicado sobre data/train.csv")
    print(f"  {ESCENARIOS[numero]}")
    print("\nAhora ejecute:  python run_pipeline.py --sin-preparar")
    print("Para restaurar: python romper_datos.py --restaurar")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sabotaje controlado del dataset")
    parser.add_argument("escenario", nargs="?", type=int, help="numero de escenario (1-7)")
    parser.add_argument("--listar", action="store_true", help="lista los escenarios")
    parser.add_argument("--restaurar", action="store_true", help="restaura los datos originales")
    args = parser.parse_args()

    if args.restaurar:
        return restaurar()
    if args.listar or args.escenario is None:
        print("Escenarios disponibles:")
        for numero, descripcion in ESCENARIOS.items():
            print(f"  {numero}. {descripcion}")
        return 0
    return aplicar(args.escenario)


if __name__ == "__main__":
    sys.exit(main())
