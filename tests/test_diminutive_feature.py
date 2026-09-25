"""Pruebas de la característica de diminutivos afectivos (diminutive_count) con
tweets reales del train de TASS 2018.

Ejecutar:  pytest tests/test_diminutive_feature.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logic.diminutive_feature import DiminutiveFeature, FEATURE_NAMES  # noqa: E402

TRAIN = pd.read_csv(ROOT / "data" / "tass" / "tass2018_es_train.csv").set_index("tweetid")
FE = DiminutiveFeature()


def value(tweetid):
    return FE.transform([TRAIN.loc[tweetid, "content"]])[0][0]


def test_vector_tiene_1_columna():
    assert FE.transform(["hola"]).shape == (1, 1)
    assert FEATURE_NAMES == ["diminutive_count"]


@pytest.mark.parametrize("tweetid, expected", [
    (768056902655864832, 1.0),   # "...un pajarito que ahora mismo..."
    (768561501611360256, 1.0),   # "...quedarse calladito..."
    (767846853459181569, 1.0),   # "...ya estoy un poquito mejor..."
    (768213876278165504, 0.0),   # tweet sin diminutivos
])
def test_diminutive_count(tweetid, expected):
    assert value(tweetid) == expected


def test_excluye_falsos_positivos():
    # verbos, adjetivos y topónimos que terminan en -ito/-illa NO son diminutivos
    assert DiminutiveFeature.count_diminutives("necesito una pastilla en Sevilla") == 0.0
    assert DiminutiveFeature.count_diminutives("es un tipo bonito e infinito") == 0.0


def test_detecta_lista_y_regla():
    # 'besitos' (lista) + 'trocito' (regla -ito) = 2 ; 'bonito' excluido
    assert DiminutiveFeature.count_diminutives("un trocito y muchos besitos, que bonito") == 2.0


def test_no_es_constante_en_el_train():
    X = FE.transform(TRAIN["content"].tolist())
    assert X.std() > 0
