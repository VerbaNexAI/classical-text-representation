"""Pruebas de la característica de léxico malsonante (vulgarity_count) con tweets
reales del train de TASS 2018.

Autor: Arley Rativa.
Ejecutar:  pytest tests/test_vulgarity_feature.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logic.vulgarity_feature import VulgarityFeature, FEATURE_NAMES  # noqa: E402

TRAIN = pd.read_csv(ROOT / "data" / "tass" / "tass2018_es_train.csv").set_index("tweetid")
FE = VulgarityFeature()


def value(tweetid):
    return FE.transform([TRAIN.loc[tweetid, "content"]])[0][0]


def test_vector_tiene_1_columna():
    assert FE.transform(["hola"]).shape == (1, 1)
    assert FEATURE_NAMES == ["vulgarity_count"]


@pytest.mark.parametrize("tweetid, expected", [
    (768213567418036224, 1.0),   # "...que puto mal escribo..."
    (768057277865717761, 1.0),   # "...el puto hueso me duele..."
    (768530150002716672, 2.0),   # "...me suelta la puta subnormal..." (puta + subnormal)
    (768213876278165504, 0.0),   # tweet sin léxico malsonante
])
def test_vulgarity_count(tweetid, expected):
    assert value(tweetid) == expected


def test_reconoce_elongaciones():
    # 'putooo' y 'joderrr' deben contar como 'puto' y 'joder'
    assert VulgarityFeature.count_vulgar("joderrr el putooo hueso") == 2.0


def test_ignora_menciones():
    # un nombre de usuario no debe contar aunque contenga una subcadena malsonante
    assert VulgarityFeature.count_vulgar("@putolobo hola que tal") == 0.0


def test_no_es_constante_en_el_train():
    X = FE.transform(TRAIN["content"].tolist())
    assert X.std() > 0
