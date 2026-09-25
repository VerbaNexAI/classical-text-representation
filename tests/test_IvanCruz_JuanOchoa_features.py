"""
Pruebas unitarias de logic/IvanCruz_JuanOchoa_features.py
Ejecutar con:  pytest -v tests/test_IvanCruz_JuanOchoa_features.py

Los tweets de la sección REAL provienen del CORPUS DE ENTRENAMIENTO de TASS 2018
(es), cada uno identificado por su ``tweetid``. ``test_real_tweets_match_train_csv``
verifica que coinciden con el CSV cuando está disponible.
"""
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "logic"))

from IvanCruz_JuanOchoa_features import (   # noqa: E402
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_SPECS,
    LexicalFeaturizer,
    audit_features,
    extract_features,
)


# ---------------------------------------------------------------------------
# Tweets REALES de train  {nombre: (tweetid, texto)}
# ---------------------------------------------------------------------------
REAL = {
    "elong_siiii": (
        768231706746912769,
        "11. siiii fue super gracioso teniamos que habernos sacado una foto"
    ),
    "elong_mooola": (
        768510568642842628,
        "@LibrosdeMaria no sé si hubo más gente, pero ayer te tagueé yo  ayy, Hannibal es que moooooola tanto!"
    ),
    "elong_TAAAAN": (
        768597310351765504,
        "Mira que soy pesimista a maza, pero odio que la gente sea TAAAAN pesimista"
    ),
    "sigla_CCC": (
        768508856301457408,
        "@Lahierritos pues sabran més q jo    *curso CCC de mirones y mironas"
    ),
    "mayus_NOHAY": (
        770247034574176256,
        "Repito, de momento NO HAY FECHA LÍMITE y las preguntas son EN EL E-MAIL no encima del vídeo"
    ),
    "excl_4": (
        767731358739402752,
        "@LuisFBonilla @isabelsevilla9 @adirecto @peichsantana Ole los buenos almerienses!!!! Mil besitos!!!!"
    ),
    "excl_3_risa": (
        768544990595284992,
        "@RoyalDarjeeling Muy mal!!! jajaja mas Geralt de rivia y menos juego de tronos"
    ),
    "quest_2": (
        768019941920473088,
        "@sitgesfestival Genial!! Vendrá @RachelNichols1 ?? Saludos y buen día!"
    ),
    "interrobang": (
        769983928841166848,
        "@EvaGelCot DONDE LO VES!!? Es mi programa favorito 😢 y sólo encuentro hasta temporada 5-6 y esos ya los tengo vistos"
    ),
    "mod_muy_demasiado": (
        768055614413086720,
        "@angeleta1973 @elespanolcom No he visto su DNI o Pasaporte pero presume de serlo aunque es muy,demasiado,peculiar y poco caritativa"
    ),
    "mod_36_y_medio": (
        768016038126620672,
        "Gracias por los ánimos! He dormido un poco más, me he tomado una pastilla y ha bajado a 36 y medio, pero aún me noto rara"
    ),
    "mod_casi": (
        768544186853359616,
        "Ahora que estoy en Madrid casi no puedo hacer fotos chachis, solo a mi careto y no"
    ),
    "interj_ah": (
        768220253730009091,
        "@Yulian_Poe @guillermoterry1 Ah. mucho más por supuesto! solo que lo incluyo. Me habías entendido mal"
    ),
    "interj_vaya_verbo": (
        768057031660072960,
        "@lmg muchísimas gracias Luis  Tú también eres un ejemplo de constancia y buen trabajo, espero que os vaya todo genial!"
    ),
    "regio_mola": (
        767723670479536128,
        "@DieRaposa @DrXaverius y otros biólogos como él se sentirán complacidos.  ¡Mola mucho!"
    ),
    "regio_Che": (
        768496682082373632,
        "@AlteaLaSueca en el debate que he estado leyendo, en el blog de Silvio @citaconsilvio lo pone el gran @feliuvicente citando al Che."
    ),
    "pol_5pos": (
        770407832508301312,
        "Me encanta estar así de feliz...ojalá y esto toda la vida!!! Buenisimas noches  Gracias, te amo @juanjoatleti 😙😘"
    ),
}


def T(name):
    return REAL[name][1]


def F(name, feat):
    return extract_features(T(name))[feat]


def f(text, feat):
    return extract_features(text)[feat]


def test_real_tweets_match_train_csv():
    default = (
        pathlib.Path(__file__).resolve().parents[1]
        / "data"
        / "tass"
        / "tass2018_es_train.csv"
    )
    path = os.environ.get("TASS_TRAIN_CSV", str(default))

    if not os.path.exists(path):
        pytest.skip("CSV de train no disponible")

    df = pd.read_csv(path)
    df["t"] = (
        df["content"]
        .fillna("")
        .str.replace("\r\n", " ")
        .str.replace("\n", " ")
    )

    by_id = dict(zip(df["tweetid"], df["t"]))

    for name, (tid, text) in REAL.items():
        assert by_id[tid] == text, f"{name} ({tid}) no coincide con el CSV"


# ---------------------------------------------------------------------------
# Estructura
# ---------------------------------------------------------------------------
def test_specs_are_consistent():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 9
    assert all(s.hypothesis.strip() for s in FEATURE_SPECS)
    assert set(extract_features("hola").keys()) == set(FEATURE_NAMES)
    assert sum(len(v) for v in FEATURE_GROUPS.values()) == len(FEATURE_NAMES)
    assert set(FEATURE_GROUPS.keys()) == {
        "expresividad",
        "modificadores",
        "coloquial",
    }


@pytest.mark.parametrize(
    "bad",
    [None, "", "   ", float("nan"), "@usuario http://t.co/x"]
)
def test_robust_to_empty_or_non_string(bad):
    assert all(v == 0.0 for v in extract_features(bad).values())


# ---------------------------------------------------------------------------
# Expresividad
# ---------------------------------------------------------------------------
def test_elongated_count_real_tweets():
    assert F("elong_siiii", "elongated_count") == 1
    # "siiii"

    assert F("elong_mooola", "elongated_count") == 1
    # "moooooola" (y "ayy" NO: solo 2 letras)

    assert F("elong_TAAAAN", "elongated_count") == 1
    # elongación en mayúsculas


def test_elongated_ignores_acronyms_and_numbers():
    assert F("sigla_CCC", "elongated_count") == 0
    # regresión: "CCC" es sigla


def test_upper_ratio_real_tweets():
    assert F("mayus_NOHAY", "upper_ratio") > 0.4
    assert F("elong_TAAAAN", "upper_ratio") == pytest.approx(1 / 13)
    assert F("pol_5pos", "upper_ratio") == 0.0


def test_punctuation_runs_real_tweets():
    assert F("excl_4", "exclam_run_max") == 4
    assert F("excl_3_risa", "exclam_run_max") == 3
    assert F("quest_2", "question_run_max") == 2
    assert F("interrobang", "interrobang_count") == 1
    # "DONDE LO VES!!?"


# ---------------------------------------------------------------------------
# Modificadores
# ---------------------------------------------------------------------------
def test_intensifiers_real_tweets():
    assert F("elong_siiii", "intensifier_count") == 1
    # "super gracioso"

    assert F("mod_muy_demasiado", "intensifier_count") == 2
    # "muy,demasiado"


def test_attenuators_real_tweets():
    assert F("mod_muy_demasiado", "attenuator_count") == 1
    # "poco caritativa"

    assert F("mod_casi", "attenuator_count") == 1
    # "casi no puedo"


def test_medio_is_not_an_attenuator():
    assert F("mod_36_y_medio", "attenuator_count") == 1
    # regresión: solo "un poco"


# ---------------------------------------------------------------------------
# Coloquial
# ---------------------------------------------------------------------------
def test_interjections_real_tweets():
    assert F("elong_mooola", "interjection_count") == 1
    # "ayy"

    assert F("interj_ah", "interjection_count") == 1
    # "Ah."

    assert F("pol_5pos", "interjection_count") == 1
    # "ojalá"


def test_vaya_verb_is_not_an_interjection():
    assert F("interj_vaya_verbo", "interjection_count") == 0
    # regresión


def test_regionalisms_real_tweets():
    assert F("regio_mola", "regionalism_count") == 1
    # "¡Mola mucho!"

    assert F("elong_mooola", "regionalism_count") == 1
    # "moooooola" (elongada)


def test_el_che_is_not_a_regionalism():
    assert F("regio_Che", "regionalism_count") == 0
    # regresión: "al Che" (Guevara)


# ---------------------------------------------------------------------------
# Adaptador scikit-learn
# ---------------------------------------------------------------------------

# Estos 4 textos se mantienen específicamente para las pruebas del
# LexicalFeaturizer, que esperan una entrada de 4 muestras.
TEXTS = [
    T("mayus_NOHAY"),
    T("excl_4"),
    T("quest_2"),
    T("regio_mola"),
]


# Esta colección es independiente de TEXTS y se utiliza para la auditoría.
# Contiene ejemplos suficientes para que las 9 características presenten
# variación y no sean detectadas como constantes.
AUDIT_TEXTS = [
    T("elong_siiii"),
    T("mayus_NOHAY"),
    T("excl_4"),
    T("quest_2"),
    T("interrobang"),
    T("mod_muy_demasiado"),
    T("mod_casi"),
    T("interj_ah"),
    T("regio_mola"),
]


def test_featurizer_shape_and_dtype():
    X = LexicalFeaturizer().fit_transform(TEXTS)

    assert X.shape == (4, len(FEATURE_NAMES))
    assert X.dtype == np.float64
    assert np.isfinite(X).all()


def test_featurizer_group_selection():
    fz = LexicalFeaturizer(groups=["expresividad"]).fit(TEXTS)

    assert list(fz.get_feature_names_out()) == list(
        FEATURE_GROUPS["expresividad"]
    )

    assert (
        LexicalFeaturizer(groups=["modificadores"])
        .fit_transform(TEXTS)
        .shape
        == (4, 2)
    )

    with pytest.raises(ValueError):
        LexicalFeaturizer(groups=["no_existe"]).fit(TEXTS)


def test_featurizer_is_stateless_and_clonable():
    from sklearn.base import clone

    fz = LexicalFeaturizer(groups=["coloquial"])

    assert clone(fz).get_params() == fz.get_params()

    a = fz.fit(TEXTS).transform(TEXTS)
    b = fz.fit(["algo totalmente distinto"]).transform(TEXTS)

    # refit no cambia nada
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Auditoría
# ---------------------------------------------------------------------------
def test_audit_detects_constant_and_length_proxy():
    def toy(t):
        return {
            "always_zero": 0.0,
            "n_words": float(len(t.split())),
        }

    rep = audit_features(
        [
            "qué tal",
            "hola",
            "más o menos",
            "hola que tal estas hoy amigo mio",
        ],
        extractor=toy,
    )

    assert bool(rep.loc["always_zero", "is_constant"]) is True
    assert bool(rep.loc["n_words", "length_proxy"]) is True


def test_audit_runs_on_all_features():
    # Se utiliza AUDIT_TEXTS, no TEXTS, porque necesitamos variedad
    # suficiente para comprobar que ninguna característica sea constante.
    rep = audit_features(AUDIT_TEXTS * 5)

    assert set(rep.index) == set(FEATURE_NAMES)

    assert {
        "activation_rate",
        "is_constant",
        "corr_n_tokens",
        "length_proxy",
    } <= set(rep.columns)

    assert rep["is_constant"].sum() == 0
    assert rep["length_proxy"].sum() == 0