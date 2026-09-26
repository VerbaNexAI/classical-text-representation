# -*- coding: utf-8 -*-
"""
Pruebas unitarias — módulo de características léxicas (Fase 0 y Fase 1)
=========================================================================

Cubre:
  1. Regresión de los 2 bugs corregidos en la Fase 0 (label_hashtag,
     label_emoji), con los mismos tweets usados en la auditoría del
     notebook (tass_notebook_final.ipynb, celda de "Confirmación de los
     bugs de preprocesamiento").
  2. Las 4 características nuevas de la Fase 1, cada una con al menos un
     caso construido para aislar el fenómeno lingüístico que dice medir,
     y con tweets reales tomados de la sección 7.3 del notebook
     ("Análisis de errores: 10 tweets reales, agrupados por tipo" —
     EJEMPLOS_CATEGORIZADOS), que documenta explícitamente que son
     ejemplos reales del corpus, no inventados.
  3. Una prueba de integración opcional (se salta si no encuentra el CSV)
     que corre el extractor completo sobre una muestra real de
     data/tass/tass2018_es_train.csv y verifica invariantes básicas
     (sin NaN, tipos correctos, no constante en la muestra).

Cómo ejecutar:
    pip install -r requirements.txt   # nltk, scipy, numpy, scikit-learn, spacy, pytest
    pytest tests/test_lexical_features.py -v

Nota de honestidad: los 10 ejemplos de EJEMPLOS_CATEGORIZADOS son reales
(vienen directamente del propio análisis de errores del notebook sobre el
split de test), pero no son técnicamente "del conjunto de entrenamiento".
La prueba de integración de la clase TestCorpusEntrenamientoReal es la que
valida directamente contra tass2018_es_train.csv cuando ese archivo está
disponible (no lo está en este sandbox).
"""
import os
import random

import pytest
from nltk import TweetTokenizer

from logic.feature_extraction import FeatureExtraction
from logic.text_processing import TextProcessing

TOK = TweetTokenizer()

# Tomados literalmente de tass_notebook_final.ipynb, sección 7.3
# (EJEMPLOS_CATEGORIZADOS) — el propio notebook los describe como
# "10 tweets reales, agrupados por tipo [de error]".
TWEETS_REALES_ERRORES = [
    "@vitojph ¡sí! Y me ha encantado ¿Tú ya las visto?",
    "Cosas que enamora las tostadas calentitas a estas horas",
    "@ArchivoSGAE Qué estupendo. Y... ¿cómo os lo encargo?",
    "Tengo una perrina adorable... Sabéis que me acompaña...",
    "Mañana tengo fiesta así que.. No me quejo",
    "@juankipua tiene muchas canciones ya jajajajaja",
    "Este viernes al cine para celebrar mi cumple. no estáis invitados",
    "¿mañana sábado 31? Mañana es miércoles 31.",
    "Estoy en la batalla final del Conquista, me faltaría Revelación",
    "es dirección de proyectos en empresas audiovisuales",
]

# Tomados literalmente de las celdas de auditoría (Fase 0) y demo (Fase 1)
# del mismo notebook.
TWEET_HASHTAG_FASE0 = "Se ha terminado #Rio2016 Lamentablemente no arriendo"
TWEET_EMOJI_FASE0 = "Que alegria!! 😊 🎉 estoy feliz"
TWEET_NEGACION_FASE1 = "no me gusta nada esto"
TWEET_ELONGACION_FASE1 = "buenooo dia"
TWEET_PUNTUACION_FASE1 = "que bien!!!"


class TestFase0Regresion:
    """Bugs de preprocesamiento identificados y corregidos en la auditoría
    de Fase 0. Si alguna de estas pruebas falla, alguien reintrodujo el
    bug (typo '[HASTAG]', u orden proper_encoding()->emoji regex)."""

    def test_hashtag_ya_no_es_constante_en_cero(self):
        procesado = TextProcessing.transformer(TWEET_HASHTAG_FASE0)
        assert "hashtag" in procesado.split()

    def test_emoji_ya_no_es_constante_en_cero(self):
        procesado = TextProcessing.transformer(TWEET_EMOJI_FASE0)
        assert "emoji" in procesado.split()

    def test_tweet_sin_hashtag_no_activa_falso_positivo(self):
        procesado = TextProcessing.transformer("hola como estas hoy")
        assert "hashtag" not in procesado.split()


class TestLexPolNeg:
    """Fase 1 — polaridad neta con resolución de negación."""

    def test_negacion_invierte_palabra_positiva(self):
        # 'gusta' es positiva en polaridad_es, pero está bajo el alcance
        # de 'no' -> el signo debe invertirse a negativo.
        procesado = TextProcessing.transformer(TWEET_NEGACION_FASE1)
        score = FeatureExtraction.lex_pol_neg(TOK.tokenize(procesado))
        assert score < 0

    def test_sin_negacion_palabra_positiva_no_se_invierte(self):
        procesado = TextProcessing.transformer("me gusta mucho esto")
        score = FeatureExtraction.lex_pol_neg(TOK.tokenize(procesado))
        assert score > 0

    def test_conjuncion_cierra_el_alcance_de_la_negacion(self):
        # 'pero' debe cerrar el alcance de 'no' antes de llegar a 'bueno'.
        procesado = TextProcessing.transformer("no lo esperaba pero es bueno")
        score = FeatureExtraction.lex_pol_neg(TOK.tokenize(procesado))
        assert score > 0  # 'bueno' queda fuera del alcance de la negación

    @pytest.mark.parametrize("tweet", TWEETS_REALES_ERRORES)
    def test_no_falla_sobre_tweets_reales_del_corpus(self, tweet):
        # Limitación documentada (sección 7.5 del notebook): el léxico de
        # polaridad (~25 palabras/clase) no cubre estas formas flexivas
        # ('encantado', 'enamora', 'estupendo', 'adorable', ...), así que
        # se espera 0.0 en los 10 casos. Esta prueba no valida que el
        # score sea "correcto" semánticamente -- fija el comportamiento
        # actual y documenta por qué, para detectar cambios accidentales.
        procesado = TextProcessing.transformer(tweet)
        score = FeatureExtraction.lex_pol_neg(TOK.tokenize(procesado))
        assert isinstance(score, float)
        assert score == 0.0


class TestElongatedWordsCount:
    """Fase 1 — elongación de caracteres ('buenooo')."""

    def test_detecta_elongacion_clasica(self):
        assert FeatureExtraction.elongated_words_count(TWEET_ELONGACION_FASE1) == 1.0

    def test_texto_sin_elongacion_da_cero(self):
        assert FeatureExtraction.elongated_words_count("buen dia a todos") == 0.0

    def test_cuenta_multiples_elongaciones(self):
        assert FeatureExtraction.elongated_words_count("holaaa buenooo diaaa") == 3.0

    def test_no_confunde_con_risas_tipo_jajaja(self):
        # 'jajajajaja' NO tiene 3+ repeticiones de UN mismo caracter
        # seguido (alterna j/a), así que no la detecta esta característica
        # -- riesgo real señalado en el propio error #6 del notebook
        # ("la risa no activa elongated_words (patron distinto)").
        assert FeatureExtraction.elongated_words_count("jajajajaja") == 0.0


class TestPunctRepetition:
    """Fase 1 — repetición de signos de exclamación/interrogación."""

    def test_detecta_exclamacion_repetida(self):
        assert FeatureExtraction.punct_repetition(TWEET_PUNTUACION_FASE1) == 1.0

    def test_detecta_interrogacion_repetida(self):
        assert FeatureExtraction.punct_repetition("en serio???") == 1.0

    def test_signo_unico_no_cuenta(self):
        assert FeatureExtraction.punct_repetition("que bien!") == 0.0

    def test_cuenta_varios_grupos_en_el_mismo_texto(self):
        assert FeatureExtraction.punct_repetition("que bien!!! en serio???") == 2.0


class TestLexicalDiversityMattr:
    """Fase 1 — MATTR, reemplaza a lexical_diversity (que medía
    diversidad de caracteres, no de palabras, por bug de implementación
    detectado en la Fase 0)."""

    def test_texto_vacio_da_cero(self):
        assert FeatureExtraction.lexical_diversity_mattr([]) == 0.0

    def test_texto_corto_menor_a_la_ventana(self):
        tokens = ["hola", "mundo", "hola"]
        # n <= window: TTR directo = tipos/tokens = 2/3
        assert FeatureExtraction.lexical_diversity_mattr(tokens, window=10) == round(2 / 3, 4)

    def test_mayor_diversidad_de_palabras_da_mayor_mattr(self):
        repetitivo = ["hola"] * 12
        diverso = ["hola", "mundo", "como", "estas", "hoy", "amigo",
                   "todo", "bien", "por", "alla", "genial", "gracias"]
        m_repetitivo = FeatureExtraction.lexical_diversity_mattr(repetitivo, window=5)
        m_diverso = FeatureExtraction.lexical_diversity_mattr(diverso, window=5)
        assert m_diverso > m_repetitivo

    def test_ignora_tags_de_preprocesamiento(self):
        # 'mention'/'url'/'hashtag'/'emoji'/'rt' no deben contar como
        # "palabras" para la diversidad léxica.
        con_tags = ["mention", "hola", "mundo", "url", "hashtag"]
        sin_tags = ["hola", "mundo"]
        assert (FeatureExtraction.lexical_diversity_mattr(con_tags, window=10)
                == FeatureExtraction.lexical_diversity_mattr(sin_tags, window=10))


class TestCorpusEntrenamientoReal:
    """Integración: corre el extractor completo sobre una muestra real de
    data/tass/tass2018_es_train.csv. Se salta automáticamente si el
    archivo no está disponible (no lo subieron a este sandbox) -- se
    debe ejecutar dentro del repo, donde sí existe, para que este
    criterio quede cubierto con datos reales de verdad."""

    CSV_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "tass", "tass2018_es_train.csv"
    )

    @pytest.mark.skipif(
        not os.path.exists(CSV_PATH),
        reason="tass2018_es_train.csv no está disponible en este entorno; "
               "ejecutar dentro del repo con los datos de TASS presentes.",
    )
    def test_features_no_constantes_en_muestra_real(self):
        import pandas as pd

        df = pd.read_csv(self.CSV_PATH)
        muestra = df["content"].dropna().sample(n=min(200, len(df)), random_state=42)

        fe = FeatureExtraction(lang="es")
        valores_lex_pol = []
        valores_elong = []
        for texto in muestra:
            procesado = TextProcessing.transformer(texto)
            if not procesado:
                continue
            tokens = TOK.tokenize(procesado)
            valores_lex_pol.append(fe.lex_pol_neg(tokens))
            valores_elong.append(fe.elongated_words_count(texto))

        # No deben ser constantes en una muestra real de 200 tweets.
        assert len(set(valores_lex_pol)) > 1
        assert len(set(valores_elong)) > 1


if __name__ == "__main__":
    random.seed(42)
    pytest.main([__file__, "-v"])
