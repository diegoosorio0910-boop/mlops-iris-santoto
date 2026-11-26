"""Permite que las pruebas importen los modulos de src/ sin instalar el paquete."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
