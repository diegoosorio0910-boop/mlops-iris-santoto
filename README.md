# MLOps Iris Santoto — Pipeline estandarizado y control de calidad de datos

Pipeline de ciencia de datos reproducible que ejecuta **preparación → validación → entrenamiento → pruebas** con un solo comando, sobre el dataset Iris, con registro de experimentos en MLflow y control de calidad de datos contra un contrato explícito.

**Actividad 2.1 — Estandarización de Pipelines y Control de Calidad de Datos**
Seminario de Grado II · Especialización en Ciencia de Datos · Universidad Santo Tomás, Seccional Tunja
Autor: Diego Alejandro Osorio Amaya · Docente: Martha Susana Contreras Ortiz, PhD

---

## Tabla de contenido

1. [Qué resuelve este proyecto](#1-qué-resuelve-este-proyecto)
2. [Requisitos previos](#2-requisitos-previos)
3. [Instalación desde cero](#3-instalación-desde-cero)
4. [Ejecución con un solo comando](#4-ejecución-con-un-solo-comando)
5. [Estructura del proyecto](#5-estructura-del-proyecto)
6. [El contrato de datos](#6-el-contrato-de-datos)
7. [Lógica de validación de datos](#7-lógica-de-validación-de-datos)
8. [Pruebas automatizadas](#8-pruebas-automatizadas)
9. [Trazabilidad: MLflow y logs](#9-trazabilidad-mlflow-y-logs)
10. [Cómo comprobar que el control de calidad funciona](#10-cómo-comprobar-que-el-control-de-calidad-funciona)
11. [Resultados obtenidos](#11-resultados-obtenidos)
12. [Reproducibilidad: qué garantiza que el resultado sea idéntico](#12-reproducibilidad-qué-garantiza-que-el-resultado-sea-idéntico)
13. [Solución de problemas](#13-solución-de-problemas)
14. [Referencias](#14-referencias)

---

## 1. Qué resuelve este proyecto

En la sesión 2 del espacio académico el experimento se ejecutaba a mano: abrir el código, cambiar parámetros, correr una celda, revisar los datos por inspección visual. Ese procedimiento depende de la memoria del investigador y produce cuatro problemas conocidos:

| Problema | Consecuencia | Cómo lo resuelve este proyecto |
|---|---|---|
| Pasos en otro orden | Se entrena antes de validar | `run_pipeline.py` fija la secuencia |
| Versiones distintas de los datos | Cada persona usa "su" CSV | `src/prepare.py` regenera los datos siempre igual |
| Entrenar sin revisar | El modelo aprende de datos erróneos | `src/validate_data.py` + `data_contract.json` |
| Resultados irrepetibles | El mismo código da números distintos | Semilla fija (`random_state=42`) y registro en MLflow |

El resultado es un **pipeline único**: un flujo fijo, repetible y documentado que cualquier tercero ejecuta con `python run_pipeline.py` y obtiene exactamente los mismos números.

Principio de diseño: **fallar rápido**. Cada etapa corre como un subproceso y el orquestador inspecciona su código de salida. Si una etapa devuelve un código distinto de `0`, el pipeline se detiene ahí y propaga ese código, de modo que nunca se entrena con datos inválidos ni se aprueba un modelo que no cumple los umbrales de calidad.

```
                                        ┌──────────────┐
 Datos de   ┌──────────────────┐   sí   │  Entrenar y  │   ┌──────────┐
 entrada ──►│ Control de       ├───────►│  registrar   ├──►│ Pruebas  ├──► Modelo
            │ calidad (etapa 2)│        │  (etapa 3)   │   │(etapa 4) │     confiable
            └────────┬─────────┘        └──────────────┘   └────┬─────┘
                     │ no                                       │ falla
                     ▼                                          ▼
            Pipeline detenido                        Modelo no aprobado
            (código de salida 1)                     (código de salida 1)
```

---

## 2. Requisitos previos

| Componente | Versión usada | Nota |
|---|---|---|
| Python | 3.13 | Funciona desde 3.10 |
| pip | 24 o superior | Incluido con Python |
| Git | cualquiera | Solo para clonar el repositorio |
| `make` | opcional | Atajos del `Makefile`; en Windows no es necesario |

No se requiere GPU, Docker, ni descargar ningún archivo de datos: el dataset Iris se obtiene programáticamente desde `sklearn.datasets.load_iris`.

Dependencias declaradas en `requirements.txt` con versión fijada:

```
scikit-learn==1.9.1   pandas==3.0.5   numpy==2.5.3
joblib==1.6.0         mlflow==3.18.0  pytest==9.1.1
```

> La lógica de validación de datos (`src/validate_data.py`) **no usa ninguna de estas librerías**: está escrita únicamente con la biblioteca estándar de Python (`csv`, `json`, `argparse`, `sys`, `datetime`, `pathlib`), según lo exigido por la actividad.

---

## 3. Instalación desde cero

Estos cuatro pasos llevan de un equipo sin nada instalado a un pipeline ejecutado.

### Windows (PowerShell)

```powershell
git clone https://github.com/[TU-USUARIO-GITHUB]/mlops-iris-santoto.git
cd mlops-iris-santoto

py -3.13 -m venv .venv          # 1. Crear el entorno virtual (una sola vez)
.venv\Scripts\activate          # 2. Activarlo (cada sesión de trabajo)
pip install -r requirements.txt # 3. Instalar dependencias (una sola vez)
python run_pipeline.py          # 4. Ejecutar el pipeline completo
```

Si PowerShell bloquea la activación del entorno:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Linux / macOS

```bash
git clone https://github.com/[TU-USUARIO-GITHUB]/mlops-iris-santoto.git
cd mlops-iris-santoto

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

### Verificar la instalación

```bash
python src/validate_data.py --ejemplos   # debe aceptar el ejemplo válido y rechazar el inválido
pytest -v                                # debe reportar 10 passed
```

---

## 4. Ejecución con un solo comando

El proyecto ofrece dos entradas equivalentes: el **script de orquestación** (multiplataforma) y el **Makefile** (atajo en sistemas con `make`).

```bash
python run_pipeline.py      # script de orquestación
make all                    # equivalente con Makefile
```

### Opciones del orquestador

| Comando | Qué hace |
|---|---|
| `python run_pipeline.py` | Ejecuta las 4 etapas en orden |
| `python run_pipeline.py --etapa 2` | Ejecuta solo la etapa indicada (1 a 4) |
| `python run_pipeline.py --solo-validar` | Solo el control de calidad |
| `python run_pipeline.py --sin-preparar` | Omite la etapa 1 y usa los CSV existentes |
| `python run_pipeline.py --sin-pruebas` | Omite las pruebas automatizadas |

### Objetivos del Makefile

| Comando | Qué hace |
|---|---|
| `make help` | Lista todos los objetivos |
| `make install` | Instala las dependencias |
| `make all` | Pipeline completo |
| `make prepare` / `validate` / `train` / `test` | Etapas 1 / 2 / 3 / 4 |
| `make contrato` | Valida los ejemplos válido e inválido del contrato |
| `make romper N=3` | Aplica un sabotaje controlado al dataset |
| `make restaurar` | Devuelve los datos originales |
| `make mlflow` | Abre la interfaz de MLflow |
| `make clean` | Elimina datos, modelos, logs y caches generados |

### Salida esperada

```
========================================================================
PIPELINE MLOPS - IRIS | Seminario de Grado II | Actividad 2.1
========================================================================
ETAPA 1: Preparacion de datos                 -> train.csv 120 filas | test.csv 30 filas
ETAPA 2: Validacion y control de calidad      -> VALIDO: cumple las 8 reglas del contrato
ETAPA 3: Entrenamiento y registro con MLflow  -> Accuracy prueba: 0.9667
ETAPA 4: Pruebas automatizadas                -> 10 passed
========================================================================
PIPELINE FINALIZADO CORRECTAMENTE
========================================================================
```

---

## 5. Estructura del proyecto

Cada archivo tiene **una sola responsabilidad**: ajustar una regla de validación no obliga a tocar el entrenamiento, y cambiar un hiperparámetro no afecta la preparación de datos.

```
mlops-iris-santoto/
├── run_pipeline.py          Orquestador: ejecuta las 4 etapas con un solo comando
├── Makefile                 Atajos equivalentes (make all, make validate, ...)
├── data_contract.json       CONTRATO DE DATOS: tipos, nulos, rangos, dominios y ejemplos
├── requirements.txt         Dependencias con versión fijada
├── romper_datos.py          Utilidad académica: sabotea el dataset para probar el control de calidad
├── README.md                Este documento
├── .gitignore               Excluye entornos, caches y artefactos regenerables
│
├── src/
│   ├── config.py            Rutas, semilla e hiperparámetros en un único lugar
│   ├── prepare.py           ETAPA 1 · carga Iris y divide 80/20 estratificado
│   ├── validate_data.py     ETAPA 2 · valida los datos contra el contrato (solo stdlib)
│   └── train.py             ETAPA 3 · entrena Random Forest y registra el run en MLflow
│
├── tests/
│   ├── conftest.py          Hace importables los módulos de src/
│   └── test_model.py        ETAPA 4 · 10 pruebas sobre datos, modelo y desempeño
│
├── docs/evidencias/         Salidas de consola de las ejecuciones válida e inválida
│
├── data/                    (generado) raw/iris.csv, train.csv, test.csv
├── models/                  (generado) model.pkl, metrics.json
├── logs/                    (generado) pipeline_*.log, validacion_*.json
├── mlruns/ · mlflow.db      (generado) trazabilidad de experimentos de MLflow
```

Las carpetas marcadas como *(generado)* están en `.gitignore`: no se versionan porque el pipeline las reconstruye por completo con un solo comando. Esa es la prueba práctica de que el proyecto es reproducible.

### Responsabilidad de cada archivo

| Archivo | Entrada | Salida | Código de salida |
|---|---|---|---|
| `run_pipeline.py` | — | Orquesta y registra el log de ejecución | 0 o el de la etapa que falló |
| `src/prepare.py` | `sklearn.datasets.load_iris` | `data/raw/iris.csv`, `train.csv`, `test.csv` | 0 / 1 |
| `src/validate_data.py` | CSV o registro JSON + `data_contract.json` | Reporte en consola + `logs/validacion_*.json` | 0 válido / 1 inválido |
| `src/train.py` | `train.csv`, `test.csv` | `models/model.pkl`, `models/metrics.json`, run en MLflow | 0 / 1 |
| `tests/test_model.py` | Datos y modelo generados | Reporte de pytest | 0 si las 10 pruebas pasan |

---

## 6. El contrato de datos

`data_contract.json` es el acuerdo formal sobre cómo deben ser los datos. Es la **única fuente de verdad**: tanto la validación previa al entrenamiento como la futura inferencia (serving con FastAPI, semana 3) leen este mismo archivo, lo que evita el *training-serving skew*.

### Esquema por columna

| Columna | Tipo | Nulos | Rango | Dominio | Ámbito |
|---|---|---|---|---|---|
| `sepal_length` | number | no | 4.0 – 8.0 | — | entrenamiento + inferencia |
| `sepal_width` | number | no | 2.0 – 4.5 | — | entrenamiento + inferencia |
| `petal_length` | number | no | 1.0 – 7.0 | — | entrenamiento + inferencia |
| `petal_width` | number | no | 0.1 – 2.5 | — | entrenamiento + inferencia |
| `species` | integer | no | 0 – 2 | {0, 1, 2} | solo entrenamiento |

El campo `scope` distingue los dos momentos del ciclo de vida: `species` es obligatoria en los archivos de entrenamiento y prueba, pero **no** en inferencia, donde es la salida del modelo y no una entrada. Las unidades son centímetros, por eso una medición de 51 cm (un registro digitado en milímetros) incumple el contrato.

### Umbrales de calidad (`quality_gates`)

```json
"min_accuracy_test": 0.8,
"max_gap_train_test_accuracy": 0.15,
"train_test_overlap_allowed": 0,
"min_rows_per_class_train": 5
```

### Ejemplo de entrada válida

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Resultado esperado: **aceptado**, 0 errores, código de salida `0`. El registro pasa al modelo, que predice `species = 0` (*setosa*).

### Ejemplo de entrada inválida

```json
{
  "sepal_length": "cinco",
  "sepal_width": null,
  "petal_length": 51.0,
  "petal_width": 0.2
}
```

Resultado esperado: **rechazado**, código de salida `1`, con tres incumplimientos simultáneos:

```
- columna 'sepal_length' [type]: el valor 'cinco' no corresponde al tipo number
- columna 'sepal_width' [nullable]: la columna no admite valores nulos
- columna 'petal_length' [maximum]: el valor 51.0 supera el maximo permitido 7.0
```

Para comprobar ambos casos:

```bash
python src/validate_data.py --ejemplos
```

---

## 7. Lógica de validación de datos

`src/validate_data.py` recorre el archivo fila por fila y contrasta cada valor con el contrato. Está implementado **sin librerías externas**: la conversión de tipos se hace con `float()` e `int()` dentro de un control de errores que reporta el problema en lugar de romper el proceso, y la lectura del CSV con `csv.DictReader`.

### Las ocho reglas

| Regla | Qué verifica | Severidad |
|---|---|---|
| **R1 Estructura** | Todas las columnas requeridas están presentes y no hay columnas extra | error |
| **R2 Nulos** | Ninguna columna con `nullable: false` admite vacíos (`""`, `NA`, `null`, `NaN`) | error |
| **R3 Tipos** | `number` acepta enteros y decimales; `integer` rechaza decimales y texto | error |
| **R4 Rangos** | Cada valor numérico está entre `minimum` y `maximum` | error |
| **R5 Dominio** | `species` solo admite los valores de `allowed_values` | error |
| **R6 Duplicados** | Reporta filas idénticas repetidas | advertencia |
| **R7 Volumen** | El archivo supera `expected_row_count_min` filas | error |
| **R8 Representatividad** | Cada clase tiene al menos `min_rows_per_class_train` filas en entrenamiento | error |

Las reglas R1 a R5 son las exigidas por la actividad (tipos, nulos y rangos lógicos). R6 a R8 son aportes propios que endurecen el control de calidad.

### Modelo de severidades

No todo hallazgo debe detener un pipeline. El contrato declara dos niveles:

- **error** → detiene el pipeline, código de salida `1`.
- **advertencia** → se reporta y se registra en el log, el pipeline continúa, código de salida `0`.

El caso concreto que motivó esta distinción: el dataset Iris original contiene **dos filas idénticas documentadas** (`5.8, 2.7, 5.1, 1.9, species=2`). Tratar los duplicados como error haría que el pipeline se detuviera siempre, incluso con los datos legítimos. Declararlos como advertencia conserva la trazabilidad del hallazgo sin producir un falso positivo que bloquee el flujo.

### Modos de ejecución

```bash
python src/validate_data.py                              # valida train.csv y test.csv
python src/validate_data.py --archivo data/train.csv     # valida un archivo específico
python src/validate_data.py --ejemplos                   # autoverifica el contrato
python src/validate_data.py --registro '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

El modo `--registro` valida un único registro con el ámbito de **inferencia**: es exactamente la verificación que ejecutará el endpoint de predicción, usando el mismo contrato y el mismo código.

### Formato del reporte

El mensaje identifica fila, columna, regla incumplida y valor problemático, para que la corrección sea inmediata:

```
Validacion fallida para train.csv: 1 error(es)
  - Fila 3, columna 'sepal_length' [type]: el valor 'cinco' no corresponde al tipo number
```

---

## 8. Pruebas automatizadas

La validación de datos y las pruebas del modelo son dos controles distintos en dos momentos distintos:

| | Validación de datos | Pruebas automatizadas |
|---|---|---|
| **Cuándo** | Antes de entrenar | Después de entrenar |
| **Qué revisa** | Los datos de entrada | Archivos, modelo y desempeño |
| **Contra qué** | El contrato de datos | Los `quality_gates` del contrato |
| **Dónde** | `src/validate_data.py` | `tests/test_model.py` |
| **Si falla** | No se entrena | El modelo no se aprueba |

Las diez pruebas de `tests/test_model.py`:

| # | Prueba | Qué garantiza |
|---|---|---|
| 01 | `archivos_procesados_existen` | La etapa 1 generó `train.csv` y `test.csv` |
| 02 | `archivos_no_estan_vacios` | Ambos archivos contienen filas |
| 03 | `columnas_coinciden_con_el_contrato` | El esquema es el declarado |
| 04 | `sin_valores_nulos` | No quedaron vacíos tras la preparación |
| 05 | `variable_objetivo_con_dominio_valido` | `species` existe y solo tiene clases permitidas |
| 06 | `archivo_del_modelo_existe` | El artefacto quedó guardado para inferencia |
| 07 | `modelo_se_carga_y_predice` | El `.pkl` se deserializa y predice |
| 08 | `accuracy_supera_umbral_minimo` | Accuracy de prueba ≥ 0.80 |
| 09 | `sin_fuga_de_datos_entre_train_y_test` | Ninguna fila de prueba está en entrenamiento |
| 10 | `brecha_train_test_no_indica_sobreajuste` | Diferencia de accuracy ≤ 0.15 |

Las pruebas 09 y 10 atienden un vacío que ninguna regla del contrato puede cubrir. Un registro de prueba copiado en entrenamiento cumple tipos, nulos y rangos, pero infla el accuracy porque **el modelo recuerda en lugar de predecir**. La prueba 09 compara los conjuntos de filas y detecta la fuga; la prueba 10 contrasta el accuracy de entrenamiento contra el de prueba, de modo que un desempeño sospechosamente perfecto se convierte en una alerta automatizada y no en una conclusión.

Ejecución aislada de las pruebas:

```bash
pytest -v                                      # las 10
pytest -v -k "fuga or sobreajuste"             # solo las de integridad experimental
```

---

## 9. Trazabilidad: MLflow y logs

### MLflow

Cada ejecución de la etapa 3 registra un *run* con parámetros (`n_estimators`, `max_depth`, `random_state`, `test_size`, versión del contrato), métricas (accuracy de entrenamiento y prueba, F1 macro, media y desviación de la validación cruzada, brecha entrenamiento-prueba) y el artefacto del modelo.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
make mlflow                                            # equivalente
```

El experimento se llama `iris-pipeline-semana2` y el almacén es un archivo SQLite local (`mlflow.db`), lo que mantiene el proyecto autocontenido y continúa la configuración usada en la Actividad 1.1.

### Logs de ejecución

El pipeline no solo corre: deja huella de qué pasó.

| Archivo | Contenido |
|---|---|
| `logs/pipeline_AAAAMMDD_HHMMSS.log` | Etapas ejecutadas, resultado y duración de cada una, código de salida final |
| `logs/validacion_AAAAMMDD_HHMMSS.json` | Reporte estructurado de la validación: errores y advertencias por archivo, fila, columna y regla |
| `logs/validacion_ultima.json` | Copia del último reporte, para inspección rápida |
| `models/metrics.json` | Métricas, hiperparámetros, semilla, versión de Python y sistema operativo |

Esta trazabilidad es la que permite demostrar, en una defensa de tesis o una auditoría, que el modelo no solo es estadísticamente válido sino técnicamente confiable.

---

## 10. Cómo comprobar que el control de calidad funciona

Un control de calidad que nunca se ha visto fallar no está verificado. `romper_datos.py` inyecta errores típicos en `data/train.csv` (guardando antes una copia en `data/_backup/`) para evidenciar en qué etapa se detiene el pipeline.

```bash
python romper_datos.py --listar      # ver los escenarios
python romper_datos.py 1             # aplicar el escenario 1
python run_pipeline.py --sin-preparar
python romper_datos.py --restaurar   # volver a los datos originales
```

Resultados verificados:

| # | Sabotaje | Etapa que lo detiene | Evidencia |
|---|---|---|---|
| 1 | `"cinco"` en vez de un número | 2 · Validación | `[type]` no corresponde al tipo number |
| 2 | Celda vacía en `petal_width` | 2 · Validación | `[nullable]` no admite nulos |
| 3 | Registro digitado en milímetros | 2 · Validación | `[maximum]` 4 valores fuera de rango |
| 4 | `species = 3` | 2 · Validación | `[allowed_values]` fuera del dominio |
| 5 | Sin flores *virginica* | 2 · Validación | `[representatividad]` la clase 2 tiene 0 filas |
| 6 | 40 % de etiquetas erradas | 4 · Pruebas | Brecha de 0.20 > 0.15 (prueba 10) |
| 7 | Filas de prueba copiadas en entrenamiento | 4 · Pruebas | 30 filas compartidas (prueba 09) |

Dos observaciones sobre este resultado:

- El escenario 5 se detiene en la **validación** y no en las pruebas, gracias a la regla R8 de representatividad: el pipeline no gasta tiempo entrenando un modelo que ya se sabe que no podrá aprender la clase ausente.
- El escenario 7 no incumple ninguna regla del contrato y produce métricas excelentes. Solo la prueba 09 lo detecta. Es la lección central: **un pipeline detecta únicamente lo que alguien le enseñó a revisar**.

---

## 11. Resultados obtenidos

Ejecución de referencia (`python run_pipeline.py`, Python 3.13, semilla 42):

| Métrica | Valor |
|---|---|
| Filas de entrenamiento / prueba | 120 / 30 |
| Distribución por clase en entrenamiento | 40 / 40 / 40 |
| Accuracy entrenamiento | 0.9833 |
| **Accuracy prueba** | **0.9667** |
| F1 macro prueba | 0.9666 |
| Validación cruzada 5-fold | 0.9583 (± 0.0000) |
| Brecha entrenamiento-prueba | 0.0167 |
| Reglas del contrato cumplidas | 8 de 8 (1 advertencia registrada) |
| Pruebas automatizadas | 10 de 10 |
| Duración total del pipeline | ≈ 10 s |

El accuracy de 0.9667 supera el umbral de 0.80 y la brecha de 0.0167 frente al entrenamiento indica que el modelo generaliza en lugar de memorizar. Las salidas de consola completas están en `docs/evidencias/`.

---

## 12. Reproducibilidad: qué garantiza que el resultado sea idéntico

| Mecanismo | Implementación |
|---|---|
| Semilla única | `RANDOM_STATE = 42` en `src/config.py`, usada en la división y en el modelo |
| División determinista | `train_test_split` estratificado con `test_size=0.2` y semilla fija |
| Dependencias fijadas | Versiones exactas en `requirements.txt` |
| Entorno aislado | Entorno virtual `.venv` documentado en la instalación |
| Configuración centralizada | Un solo archivo (`src/config.py`) define rutas, semilla e hiperparámetros |
| Datos regenerables | Nada depende de un CSV enviado por correo: `prepare.py` reconstruye los datos |
| Experimentos registrados | Parámetros, métricas y artefactos en MLflow |
| Entorno documentado | `models/metrics.json` guarda versión de Python y sistema operativo |
| Verificación automática | 10 pruebas que fallan si el resultado se degrada |

Prueba de reproducibilidad: `make clean && make all` elimina todos los artefactos generados y los reconstruye. El accuracy reportado debe ser idéntico (0.9667).

---

## 13. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `No such file or directory: data/train.csv` | Se omitió la etapa 1 | `python run_pipeline.py` (sin `--sin-preparar`) |
| `ModuleNotFoundError: No module named 'sklearn'` | Entorno sin dependencias o sin activar | Activar `.venv` y `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'config'` | Se ejecutó un módulo desde otra carpeta | Ejecutar siempre desde la raíz del proyecto |
| El pipeline se detiene en la etapa 2 | Los datos incumplen el contrato | Revisar el reporte y `logs/validacion_ultima.json`; si viene de un sabotaje, `python romper_datos.py --restaurar` |
| `AVISO: MLflow no esta instalado` | Falta la dependencia | `pip install -r requirements.txt`; el modelo y las métricas locales se guardan igual |
| PowerShell no activa el entorno | Política de ejecución restringida | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| `make: command not found` | `make` no está instalado (habitual en Windows) | Usar `python run_pipeline.py`, que es equivalente |
| El puerto 5000 está ocupado | Otra instancia de MLflow | `mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db` |

---

## 14. Referencias

Contreras Ortiz, M. S., Puerto Moreno, S. A., Suárez Rojas, A. L., & Vela Beltrán, D. A. (2026). *Seminario de Grado II: Métodos de investigación en Data Science*. Universidad Santo Tomás, Seccional Tunja.

Mensh, B., & Kording, K. (2017). Ten simple rules for structuring papers. *PLoS Computational Biology, 13*(9), e1005619. https://doi.org/10.1371/journal.pcbi.1005619

MLOps Open Source - Santoto. (2026). *Portal del curso MLOps: Semana 2, pipelines y calidad*. https://dialvebe.github.io/mlops-course-santoto/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program). *Journal of Machine Learning Research, 22*(164), 1–20.

---

## Licencia y uso académico

Proyecto desarrollado con fines académicos para el espacio académico Seminario de Grado II de la Especialización en Ciencia de Datos de la Universidad Santo Tomás, Seccional Tunja. El dataset Iris proviene de `scikit-learn` y es de dominio público.
