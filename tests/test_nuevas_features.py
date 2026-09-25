"""Pruebas de emoji_pol, laugh_count y lex_pol_neg con tweets reales del train de TASS 2018."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logic.feature_extraction import FeatureExtraction, NEW_FEATURE_NAMES  # noqa: E402

TRAIN = pd.read_csv(ROOT / 'data' / 'tass' / 'tass2018_es_train.csv').set_index('tweetid')
FE = FeatureExtraction('es')


def features(tweetid):
    """Caracteristicas nuevas de un tweet real del train, como diccionario."""
    return dict(zip(NEW_FEATURE_NAMES, FE.get_features_nuevas(TRAIN.loc[tweetid, 'content'])))


def test_vector_tiene_una_columna_por_caracteristica():
    assert FE.get_features_nuevas('hola').shape == (len(NEW_FEATURE_NAMES),)


# --- emoji_pol ---------------------------------------------------------------------------
@pytest.mark.parametrize('tweetid, count, pol', [
    (771086980373417984, 1, 1),    # "...para que te sientas mejor❤"
    (770279138146017280, 1, -1),   # "...me duele la cabeza que está que me explota😞"
    (770371876753403904, 5, 5),    # "...menos mal que no has dicho escuchal 😂😂😂😂😂"
    (769615173141295104, 0, 0),    # sin emojis
])
def test_emoji_pol(tweetid, count, pol):
    f = features(tweetid)
    assert (f['emoji_count'], f['emoji_pol']) == (count, pol)


# --- laugh_count -------------------------------------------------------------------------
@pytest.mark.parametrize('tweetid, count, length', [
    (768213567418036224, 1, 8),    # "...ha quedado raro el "cómetelo" ahí JAJAJAJA"
    (768058983827537920, 1, 5),    # "Ayer me quedé dormido xdddd..."
    (770300836480163840, 1, 2),    # "me pasaba igual xD, ..."
    (771086980373417984, 0, 0),    # sin risas
])
def test_laugh_count(tweetid, count, length):
    f = features(tweetid)
    assert (f['laugh_count'], f['laugh_len']) == (count, length)


# --- lex_pol_neg -------------------------------------------------------------------------
def test_lex_pol_neg_positivo_sin_negacion():
    # "...para que te sientas mejor❤" -> 'mejor' (+0.625)
    assert features(771086980373417984)['lex_pol_neg'] == pytest.approx(0.625)


def test_lex_pol_neg_negativo():
    # "...que puto mal escribo..." -> 'puto' (-0.375) y 'mal' (-0.525)
    assert features(768213567418036224)['lex_pol_neg'] == pytest.approx(-0.45)


def test_lex_pol_neg_adversativa_cierra_la_negacion():
    # "Jo no me parece bien porque ..., pero bueno al menos ..." (N)
    # 'bien' queda negado (-0.833) y 'pero' corta el alcance: 'bueno' sigue positivo (+0.833)
    assert features(770403844282953729)['lex_pol_neg'] == pytest.approx(0.0)


def test_lex_pol_neg_la_negacion_invierte_el_signo():
    positivo = FE.get_features_nuevas('es bueno')[4]
    negado = FE.get_features_nuevas('no es bueno')[4]
    assert positivo > 0 and negado == pytest.approx(-positivo)


# --- Caracteristicas de Zuly Gonzalez ------------------------------------------------------
def test_doubt_count_con_tweet_real_del_corpus():
    assert features(768228346232729603)['doubt_count'] >= 1


def test_regionalism_count_con_tweet_real_del_corpus():
    assert features(768221021300264964)['regionalism_count'] >= 1


def test_intensifier_count_con_tweet_real_del_corpus():
    assert features(768213876278165504)['intensifier_count'] >= 1


# --- Caracteristicas de Daniel Contreras ---------------------------------------------------
def test_social_pos_con_tweet_real_del_corpus():
    # "Gracias por los ánimos! He dormido un poco más..." -> 'gracias' y 'ánimos'
    assert features(768016038126620672)['social_pos'] == 2


def test_groserias_con_tweet_real_del_corpus():
    # "...he notado como el puto hueso se movía" -> 'puto'
    assert features(768057277865717761)['groserias'] == 1


def test_groserias_sin_tildes_ni_mayusculas():
    assert FE.get_features_nuevas('Qué ASCO, imbécil')[NEW_FEATURE_NAMES.index('groserias')] == 2
