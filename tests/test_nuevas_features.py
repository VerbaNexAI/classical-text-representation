"""Pruebas de emoji_pol, laugh_count y lex_pol_neg con tweets reales del train de TASS 2018."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logic.feature_extraction import FeatureExtraction, NEW_FEATURE_NAMES

TRAIN = pd.read_csv(ROOT / 'data' / 'tass' / 'tass2018_es_train.csv').set_index('tweetid')
FE = FeatureExtraction('es')


def features(tweetid):
    """Caracteristicas nuevas de un tweet real del train, como diccionario."""
    return dict(zip(NEW_FEATURE_NAMES, FE.get_features_nuevas(TRAIN.loc[tweetid, 'content'])))


def test_vector_tiene_las_8_columnas():
    assert FE.get_features_nuevas(TRAIN.loc[771086980373417984, 'content']).shape == (8,)


@pytest.mark.parametrize('tweetid, count, pol', [
    (771086980373417984, 1, 1),
    (770279138146017280, 1, -1),
    (770371876753403904, 5, 5),
    (769615173141295104, 0, 0),
])
def test_emoji_pol(tweetid, count, pol):
    f = features(tweetid)
    assert (f['emoji_count'], f['emoji_pol']) == (count, pol)


@pytest.mark.parametrize('tweetid, count, length', [
    (768213567418036224, 1, 8),
    (768058983827537920, 1, 5),
    (770300836480163840, 1, 2),
    (771086980373417984, 0, 0),
])
def test_laugh_count(tweetid, count, length):
    f = features(tweetid)
    assert (f['laugh_count'], f['laugh_len']) == (count, length)


def test_lex_pol_neg_positivo_sin_negacion():
    assert features(771086980373417984)['lex_pol_neg'] == pytest.approx(0.625)


def test_lex_pol_neg_negativo():
    assert features(768213567418036224)['lex_pol_neg'] == pytest.approx(-0.45)


def test_lex_pol_neg_adversativa_cierra_la_negacion():
    assert features(770403844282953729)['lex_pol_neg'] == pytest.approx(0.0)


def test_lex_pol_neg_la_negacion_invierte_el_signo():
    assert features(768231166965145600)['lex_pol_neg'] < 0


def test_doubt_count_con_tweet_real_del_corpus():
    assert features(768228346232729603)['doubt_count'] >= 1


def test_regionalism_count_con_tweet_real_del_corpus():
    assert features(768221021300264964)['regionalism_count'] >= 1


def test_intensifier_count_con_tweet_real_del_corpus():
    assert features(768213876278165504)['intensifier_count'] >= 1
