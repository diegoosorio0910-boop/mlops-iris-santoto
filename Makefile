# Makefile - MLOps Iris Santoto | Seminario de Grado II, Actividad 2.1
# Atajos equivalentes al orquestador run_pipeline.py.
# Uso: make all   (o make help para ver los objetivos disponibles)

PYTHON ?= python

.PHONY: help setup install prepare validate train test all pipeline contrato romper restaurar mlflow clean

help:  ## Muestra los objetivos disponibles
	@echo "Objetivos disponibles:"
	@echo "  make install    Instala las dependencias de requirements.txt"
	@echo "  make all        Ejecuta el pipeline completo (preparar, validar, entrenar, probar)"
	@echo "  make prepare    Etapa 1: preparacion de datos"
	@echo "  make validate   Etapa 2: validacion contra el contrato de datos"
	@echo "  make train      Etapa 3: entrenamiento y registro en MLflow"
	@echo "  make test       Etapa 4: pruebas automatizadas (pytest)"
	@echo "  make contrato   Valida los ejemplos valido e invalido del contrato"
	@echo "  make romper N=3 Aplica un sabotaje controlado al dataset"
	@echo "  make restaurar  Restaura los datos originales"
	@echo "  make mlflow     Abre la interfaz de MLflow en http://127.0.0.1:5000"
	@echo "  make clean      Elimina datos, modelos, logs y caches generados"

install:  ## Instala dependencias
	$(PYTHON) -m pip install -r requirements.txt

setup: install  ## Alias de install

all:  ## Pipeline completo con un solo comando
	$(PYTHON) run_pipeline.py

pipeline: all  ## Alias de all

prepare:  ## Etapa 1
	$(PYTHON) run_pipeline.py --etapa 1

validate:  ## Etapa 2
	$(PYTHON) run_pipeline.py --etapa 2

train:  ## Etapa 3
	$(PYTHON) run_pipeline.py --etapa 3

test:  ## Etapa 4
	$(PYTHON) run_pipeline.py --etapa 4

contrato:  ## Valida los ejemplos del contrato de datos
	$(PYTHON) src/validate_data.py --ejemplos

N ?= 1
romper:  ## Sabotaje controlado: make romper N=3
	$(PYTHON) romper_datos.py $(N)

restaurar:  ## Restaura los datos originales
	$(PYTHON) romper_datos.py --restaurar

mlflow:  ## Interfaz de MLflow
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:  ## Limpia los artefactos generados
	rm -rf data/train.csv data/test.csv data/raw data/_backup models logs mlruns mlflow.db
	rm -rf .pytest_cache src/__pycache__ tests/__pycache__
	@echo "Artefactos generados eliminados. Ejecute 'make all' para regenerarlos."
