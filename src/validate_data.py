"""ETAPA 2 - Validacion de datos y control de calidad.

Responsabilidad unica: contrastar los datos contra data_contract.json ANTES de
entrenar. Si los datos no cumplen el contrato, el modelo no se entrena.

RESTRICCION ACADEMICA: este modulo NO usa librerias externas (no pandas, no
numpy, no Great Expectations, no pydantic). Solo biblioteca estandar de Python:
csv, json, argparse, sys, datetime, pathlib. De esta forma la logica de
validacion queda explicita, auditable y portable.

Reglas implementadas (declaradas en el contrato):
  R1 Estructura        columnas requeridas presentes y sin columnas extra
  R2 Nulos             ninguna columna con nullable=false admite vacios
  R3 Tipos             number acepta int/float; integer rechaza decimales y texto
  R4 Rangos            minimum <= valor <= maximum
  R5 Dominio           species solo admite allowed_values
  R6 Duplicados        no se admiten filas identicas repetidas
  R7 Volumen           el archivo debe superar expected_row_count_min filas
  R8 Representatividad cada clase con al menos min_rows_per_class_train filas

Uso:
  python src/validate_data.py                         # valida train.csv y test.csv
  python src/validate_data.py --archivo data/train.csv
  python src/validate_data.py --ejemplos              # valida los ejemplos del contrato
  python src/validate_data.py --registro '{"sepal_length": 5.1, ...}'

Codigos de salida: 0 = datos validos, 1 = validacion fallida
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from config import CONTRACT_FILE, LOGS_DIR, TEST_FILE, TRAIN_FILE, asegurar_directorios

NULOS = {"", "na", "n/a", "nan", "null", "none", "nulo", "vacio"}
LIMITE_ERRORES_MOSTRADOS = 15


# --------------------------------------------------------------------------- #
# Utilidades de tipo (sin librerias externas)
# --------------------------------------------------------------------------- #
def es_nulo(valor) -> bool:
    """True si el valor representa un vacio o nulo en cualquiera de sus formas."""
    if valor is None:
        return True
    return str(valor).strip().lower() in NULOS


def convertir(valor, tipo: str):
    """Intenta convertir el valor al tipo del contrato.

    Devuelve (valor_convertido, ok). No lanza excepciones: el pipeline debe
    reportar el error, no romperse por el.
    """
    texto = str(valor).strip()
    if tipo == "number":
        try:
            return float(texto), True
        except ValueError:
            return None, False
    if tipo == "integer":
        try:
            numero = float(texto)
        except ValueError:
            return None, False
        if numero != int(numero):           # 2.5 no es un entero valido
            return None, False
        return int(numero), True
    if tipo == "string":
        return texto, True
    return None, False


# --------------------------------------------------------------------------- #
# Validacion de un registro (sirve para CSV y para inferencia)
# --------------------------------------------------------------------------- #
def validar_registro(registro: dict, columnas: dict, ambito: str, fila=None) -> list[dict]:
    """Valida un unico registro contra las reglas R2-R5 del contrato.

    ambito: "training" (exige species) o "inference" (solo las 4 features).
    """
    errores: list[dict] = []

    for nombre, regla in columnas.items():
        if ambito not in regla.get("scope", ["training"]):
            continue
        if nombre not in registro:
            if regla.get("required", False):
                errores.append(_error(fila, nombre, "required",
                                      None, "la columna requerida no esta presente"))
            continue

        valor = registro[nombre]

        # R2 - Nulos
        if es_nulo(valor):
            if not regla.get("nullable", True):
                errores.append(_error(fila, nombre, "nullable", valor,
                                      "la columna no admite valores nulos"))
            continue

        # R3 - Tipos
        convertido, ok = convertir(valor, regla["type"])
        if not ok:
            errores.append(_error(fila, nombre, "type", valor,
                                  f"el valor '{valor}' no corresponde al tipo {regla['type']}"))
            continue

        # R4 - Rangos
        if "minimum" in regla and convertido < regla["minimum"]:
            errores.append(_error(fila, nombre, "minimum", convertido,
                                  f"el valor {convertido} es inferior al minimo permitido {regla['minimum']}"))
        if "maximum" in regla and convertido > regla["maximum"]:
            errores.append(_error(fila, nombre, "maximum", convertido,
                                  f"el valor {convertido} supera el maximo permitido {regla['maximum']}"))

        # R5 - Dominio de clases
        if "allowed_values" in regla and convertido not in regla["allowed_values"]:
            errores.append(_error(fila, nombre, "allowed_values", convertido,
                                  f"el valor {convertido} no pertenece al dominio {regla['allowed_values']}"))

    return errores


def _error(fila, columna, regla, valor, mensaje, severidad="error") -> dict:
    """Construye un hallazgo. severidad 'error' detiene el pipeline; 'warning' no."""
    return {"fila": fila, "columna": columna, "regla": regla,
            "valor": valor, "mensaje": mensaje, "severidad": severidad}


def solo_errores(hallazgos: list[dict]) -> list[dict]:
    return [h for h in hallazgos if h.get("severidad", "error") == "error"]


def solo_advertencias(hallazgos: list[dict]) -> list[dict]:
    return [h for h in hallazgos if h.get("severidad") == "warning"]


# --------------------------------------------------------------------------- #
# Validacion de un archivo CSV completo
# --------------------------------------------------------------------------- #
def validar_archivo(ruta: Path, contrato: dict) -> list[dict]:
    """Aplica R1-R8 sobre un archivo CSV de entrenamiento o prueba."""
    errores: list[dict] = []

    if not ruta.exists():
        return [_error(None, None, "archivo", str(ruta),
                       "el archivo no existe: ejecute primero la etapa de preparacion")]

    columnas = contrato["columns"]
    dataset = contrato["dataset"]
    requeridas = [c for c, r in columnas.items()
                  if r.get("required") and "training" in r.get("scope", ["training"])]

    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        encabezado = lector.fieldnames or []

        # R1 - Estructura
        faltantes = [c for c in requeridas if c not in encabezado]
        for columna in faltantes:
            errores.append(_error(None, columna, "estructura", None,
                                  "columna requerida ausente en el encabezado"))
        if not dataset.get("allow_extra_columns", True):
            for columna in encabezado:
                if columna not in columnas:
                    errores.append(_error(None, columna, "estructura", None,
                                          "columna no declarada en el contrato"))
        if faltantes:
            return errores  # sin columnas no tiene sentido validar fila por fila

        vistas: set[tuple] = set()
        conteo_clases: dict = {}
        total = 0

        for indice, registro in enumerate(lector, start=2):  # fila 1 = encabezado
            total += 1
            errores.extend(validar_registro(registro, columnas, "training", fila=indice))

            # R6 - Duplicados
            huella = tuple(registro.get(c) for c in encabezado)
            if not dataset.get("allow_duplicate_rows", True):
                if huella in vistas:
                    errores.append(_error(
                        indice, None, "duplicado", None,
                        "la fila esta repetida en el archivo",
                        severidad=dataset.get("duplicate_rows_severity", "error")))
                vistas.add(huella)

            clase = registro.get(contrato["target"])
            conteo_clases[clase] = conteo_clases.get(clase, 0) + 1

    # R7 - Volumen
    minimo = dataset.get("expected_row_count_min", 0)
    if total < minimo:
        errores.append(_error(None, None, "volumen", total,
                              f"el archivo tiene {total} filas y el contrato exige al menos {minimo}"))

    # R8 - Representatividad (solo en entrenamiento)
    if ruta.name == TRAIN_FILE.name:
        minimo_clase = contrato["quality_gates"].get("min_rows_per_class_train", 0)
        esperadas = columnas[contrato["target"]].get("allowed_values", [])
        for clase in esperadas:
            presentes = conteo_clases.get(str(clase), 0)
            if presentes < minimo_clase:
                errores.append(_error(None, contrato["target"], "representatividad", presentes,
                                      f"la clase {clase} solo tiene {presentes} filas y se exigen {minimo_clase}"))

    return errores


# --------------------------------------------------------------------------- #
# Reporte y trazabilidad
# --------------------------------------------------------------------------- #
def _linea(hallazgo: dict) -> str:
    ubicacion = f"Fila {hallazgo['fila']}, " if hallazgo["fila"] else ""
    columna = f"columna '{hallazgo['columna']}'" if hallazgo["columna"] else "archivo"
    return f"    - {ubicacion}{columna} [{hallazgo['regla']}]: {hallazgo['mensaje']}"


def imprimir_errores(nombre: str, errores: list[dict]) -> None:
    print(f"  Validacion fallida para {nombre}: {len(errores)} error(es)")
    for error in errores[:LIMITE_ERRORES_MOSTRADOS]:
        print(_linea(error))
    if len(errores) > LIMITE_ERRORES_MOSTRADOS:
        print(f"    ... y {len(errores) - LIMITE_ERRORES_MOSTRADOS} error(es) mas (ver el log)")


def imprimir_advertencias(nombre: str, advertencias: list[dict]) -> None:
    print(f"  {nombre}: {len(advertencias)} advertencia(s) registrada(s), no detienen el pipeline")
    for aviso in advertencias[:LIMITE_ERRORES_MOSTRADOS]:
        print(_linea(aviso))


def escribir_log(resultado: dict) -> Path:
    """Deja huella digital de que validaciones pasaron y cuales fallaron."""
    asegurar_directorios()
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = LOGS_DIR / f"validacion_{marca}.json"
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    ultimo = LOGS_DIR / "validacion_ultima.json"
    ultimo.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


# --------------------------------------------------------------------------- #
# Modos de ejecucion
# --------------------------------------------------------------------------- #
def validar_ejemplos_del_contrato(contrato: dict) -> int:
    """Autoverifica el contrato: el ejemplo valido debe pasar y el invalido fallar."""
    columnas = contrato["columns"]
    ejemplos = contrato["examples"]
    exito = 0

    print("  Caso 1 - ejemplo de entrada VALIDA del contrato")
    errores = validar_registro(ejemplos["valid_input"]["payload"], columnas, "inference")
    if errores:
        print("    RESULTADO INESPERADO: el ejemplo valido fue rechazado")
        imprimir_errores("valid_input", errores)
        exito = 1
    else:
        print("    ACEPTADO (0 errores) -> el registro puede ir al modelo")

    print("  Caso 2 - ejemplo de entrada INVALIDA del contrato")
    errores = validar_registro(ejemplos["invalid_input"]["payload"], columnas, "inference")
    if errores:
        print(f"    RECHAZADO ({len(errores)} errores) -> el modelo no se ejecuta")
        for error in errores:
            print(f"    - columna '{error['columna']}' [{error['regla']}]: {error['mensaje']}")
    else:
        print("    RESULTADO INESPERADO: el ejemplo invalido fue aceptado")
        exito = 1

    return exito


def main() -> int:
    parser = argparse.ArgumentParser(description="Validacion de datos contra data_contract.json")
    parser.add_argument("--archivo", action="append", default=None,
                        help="ruta de un CSV a validar (se puede repetir)")
    parser.add_argument("--ejemplos", action="store_true",
                        help="valida los ejemplos valido/invalido declarados en el contrato")
    parser.add_argument("--registro", default=None,
                        help="valida un unico registro de inferencia en formato JSON")
    args = parser.parse_args()

    contrato = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    print(f"  Contrato: {contrato['contract_name']} v{contrato['contract_version']}")
    print(f"  Reglas activas: {len(contrato['validation_rules_implemented'])}")

    # Modo inferencia: un registro suelto
    if args.registro:
        registro = json.loads(args.registro)
        errores = solo_errores(validar_registro(registro, contrato["columns"], "inference"))
        if errores:
            imprimir_errores("registro de inferencia", errores)
            escribir_log({"modo": "inferencia", "valido": False, "errores": errores})
            return 1
        print("  Registro ACEPTADO: cumple el contrato de inferencia")
        escribir_log({"modo": "inferencia", "valido": True, "errores": []})
        return 0

    # Modo autoverificacion del contrato
    if args.ejemplos:
        return validar_ejemplos_del_contrato(contrato)

    # Modo archivos
    rutas = [Path(a) for a in args.archivo] if args.archivo else [TRAIN_FILE, TEST_FILE]
    resultado = {"fecha": datetime.now().isoformat(timespec="seconds"),
                 "contrato": f"{contrato['contract_name']} v{contrato['contract_version']}",
                 "archivos": []}
    codigo = 0

    total_reglas = len(contrato["validation_rules_implemented"])
    for ruta in rutas:
        hallazgos = validar_archivo(ruta, contrato)
        errores = solo_errores(hallazgos)
        advertencias = solo_advertencias(hallazgos)

        resultado["archivos"].append({
            "archivo": str(ruta),
            "valido": not errores,
            "total_errores": len(errores),
            "total_advertencias": len(advertencias),
            "errores": errores,
            "advertencias": advertencias,
        })

        if errores:
            imprimir_errores(ruta.name, errores)
            codigo = 1
        else:
            print(f"  {ruta.name}: VALIDO - cumple las {total_reglas} reglas del contrato")
        if advertencias:
            imprimir_advertencias(ruta.name, advertencias)

    resultado["valido"] = codigo == 0
    log = escribir_log(resultado)
    print(f"  Log de trazabilidad: {log.name}")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
