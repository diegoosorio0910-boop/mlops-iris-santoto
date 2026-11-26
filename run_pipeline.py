#!/usr/bin/env python3
"""ORQUESTADOR DEL PIPELINE - MLOps Iris Santoto.

Ejecuta todo el flujo con un solo comando:

    python run_pipeline.py

Etapas: 1) Preparar  2) Validar  3) Entrenar y registrar  4) Probar

Principio "fallar rapido": cada etapa se ejecuta como un subproceso y se
inspecciona su codigo de salida. Si una etapa devuelve un codigo distinto de 0,
el pipeline se detiene de inmediato y propaga ese codigo, de modo que nunca se
entrena con datos invalidos ni se aprueba un modelo que no cumple los umbrales.

Opciones:
    --sin-preparar    omite la etapa 1 (usa los CSV existentes, util para demos)
    --sin-pruebas     omite la etapa 4
    --solo-validar    ejecuta unicamente la etapa 2
    --etapa N         ejecuta unicamente la etapa N (1-4)

Codigos de salida: 0 = pipeline completo, != 0 = codigo de la etapa que fallo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
LOGS = RAIZ / "logs"
ANCHO = 72

ETAPAS = [
    (1, "Preparacion de datos", [sys.executable, "src/prepare.py"]),
    (2, "Validacion y control de calidad", [sys.executable, "src/validate_data.py"]),
    (3, "Entrenamiento y registro con MLflow", [sys.executable, "src/train.py"]),
    (4, "Pruebas automatizadas", [sys.executable, "-m", "pytest", "tests/", "-v",
                                  "--tb=short", "-p", "no:cacheprovider"]),
]


def titulo(texto: str) -> None:
    print("=" * ANCHO)
    print(texto)
    print("=" * ANCHO)


def ejecutar(numero: int, nombre: str, comando: list[str], registro: list[str]) -> int:
    print()
    print("-" * ANCHO)
    print(f"ETAPA {numero}: {nombre}")
    print("-" * ANCHO)

    inicio = time.perf_counter()
    resultado = subprocess.run(comando, cwd=RAIZ)
    duracion = time.perf_counter() - inicio

    if resultado.returncode != 0:
        print()
        print(f"  >>> ERROR EN LA ETAPA {numero}: {nombre}")
        print(f"  >>> codigo de salida {resultado.returncode}. El pipeline se detiene aqui.")
        registro.append(f"ETAPA {numero} - {nombre}: FALLO (codigo {resultado.returncode}, "
                        f"{duracion:.2f}s)")
        return resultado.returncode

    print(f"\n  Etapa {numero} completada en {duracion:.2f} s")
    registro.append(f"ETAPA {numero} - {nombre}: OK ({duracion:.2f}s)")
    return 0


def guardar_log(registro: list[str], codigo: int, total: float) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = LOGS / f"pipeline_{marca}.log"
    estado = "PIPELINE FINALIZADO CORRECTAMENTE" if codigo == 0 else "PIPELINE DETENIDO POR ERROR"
    contenido = [
        f"Ejecucion: {datetime.now().isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]}",
        "",
        *registro,
        "",
        f"Duracion total: {total:.2f}s",
        f"Resultado: {estado} (codigo {codigo})",
    ]
    destino.write_text("\n".join(contenido) + "\n", encoding="utf-8")
    return destino


def main() -> int:
    parser = argparse.ArgumentParser(description="Orquestador del pipeline MLOps Iris")
    parser.add_argument("--sin-preparar", action="store_true", help="omite la etapa 1")
    parser.add_argument("--sin-pruebas", action="store_true", help="omite la etapa 4")
    parser.add_argument("--solo-validar", action="store_true", help="ejecuta solo la etapa 2")
    parser.add_argument("--etapa", type=int, choices=[1, 2, 3, 4], help="ejecuta solo una etapa")
    args = parser.parse_args()

    seleccion = list(ETAPAS)
    if args.etapa:
        seleccion = [e for e in ETAPAS if e[0] == args.etapa]
    elif args.solo_validar:
        seleccion = [e for e in ETAPAS if e[0] == 2]
    else:
        if args.sin_preparar:
            seleccion = [e for e in seleccion if e[0] != 1]
        if args.sin_pruebas:
            seleccion = [e for e in seleccion if e[0] != 4]

    titulo("PIPELINE MLOPS - IRIS | Seminario de Grado II | Actividad 2.1")
    print(f"Inicio: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Etapas programadas: {', '.join(str(e[0]) for e in seleccion)}")

    registro: list[str] = []
    inicio = time.perf_counter()

    for numero, nombre, comando in seleccion:
        codigo = ejecutar(numero, nombre, comando, registro)
        if codigo != 0:
            total = time.perf_counter() - inicio
            log = guardar_log(registro, codigo, total)
            print()
            titulo("PIPELINE DETENIDO POR ERROR")
            print(f"Log de ejecucion: logs/{log.name}")
            return codigo

    total = time.perf_counter() - inicio
    log = guardar_log(registro, 0, total)
    print()
    titulo("PIPELINE FINALIZADO CORRECTAMENTE")
    for linea in registro:
        print(f"  {linea}")
    print(f"\nDuracion total: {total:.2f} s")
    print(f"Log de ejecucion: logs/{log.name}")
    print("Para ver los experimentos:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
