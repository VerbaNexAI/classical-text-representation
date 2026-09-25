"""Característica léxica: densidad de léxico malsonante/vulgar (TASS 2018).

Autor: Arley Rativa.

Una sola característica, `vulgarity_count`: número de términos malsonantes (tacos,
insultos, palabras soeces) que aparecen en el tuit.

Hipótesis lingüística
---------------------
El léxico malsonante es un marcador pragmático de **alta intensidad emocional**,
mayoritariamente **negativa** (enfado, queja, insulto). La Bolsa de Palabras trata
cada taco como un token más y no captura que su presencia, en conjunto, es una
señal de polaridad; agruparlos en un solo rasgo la hace explícita.

Robustez (para que "no se escape" nada):
  * Se comparan las palabras en minúscula y **sin acentos**.
  * Se **eliminan menciones (@), hashtags (#) y URLs** antes de contar, para no
    confundir nombres de usuario con palabras.
  * Se **colapsan las elongaciones** de 3+ caracteres iguales ('putooo' -> 'puto',
    'joderrr' -> 'joder') para reconocer el énfasis alargado típico de Twitter.
  * El léxico incluye variantes de género/número y formas frecuentes en varias
    variedades del español.

Se implementa como un transformador compatible con scikit-learn (`fit`/`transform`),
*stateless*, para poder unirlo con BoW/TF-IDF mediante `FeatureUnion` o
`scipy.sparse.hstack` sin fuga de información.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = ["vulgarity_count"]

# Léxico malsonante curado (minúscula, sin acentos). Incluye variantes de
# género/número y algunas formas regionales.
VULGAR_LEXICON = {
    "puto", "puta", "putos", "putas", "putada", "putadas", "putisimo", "putamente",
    "mierda", "mierdas", "mierdon", "joder", "joder", "jodido", "jodida", "jodidos",
    "jodidas", "cono", "conazo", "cabron", "cabrona", "cabrones", "cabronazo",
    "gilipollas", "gilipolla", "gilipollez", "gilipolleces", "subnormal",
    "subnormales", "hostia", "hostias", "follar", "folla", "follando", "polla",
    "pollas", "capullo", "capulla", "capullos", "verga", "vergas", "zorra", "zorras",
    "perra", "perras", "pendejo", "pendeja", "pendejos", "chingar", "chinga",
    "chingada", "estupido", "estupida", "estupidos", "estupidas", "asqueroso",
    "asquerosa", "asquerosos", "imbecil", "imbeciles", "idiota", "idiotas",
    "maricon", "maricones", "marica", "cojones", "cojon", "cagar", "cagada",
    "cagadas", "cagon", "mamon", "mamona", "mamada", "gonorrea", "malparido",
    "malparida", "hijueputa", "hpta", "carajo", "coj0nes",
}

_STRIP_RE = re.compile(r"@\w+|#\w+|https?://\S+|www\.\S+")
_ELONG_RE = re.compile(r"(.)\1{2,}")       # 3+ caracteres iguales -> 1 (putooo -> puto)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

FEATURE_DOC = {
    "vulgarity_count":
        "Número de términos malsonantes en el tuit. Hipótesis: marca alta "
        "intensidad emocional, típicamente negativa (N).",
}


def _prepare(text: str) -> list:
    """Quita menciones/hashtags/URLs, pasa a minúscula sin acentos, colapsa
    elongaciones y devuelve la lista de palabras."""
    text = _STRIP_RE.sub(" ", text)
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _ELONG_RE.sub(r"\1", text)
    return _WORD_RE.findall(text)


class VulgarityFeature(BaseEstimator, TransformerMixin):
    """Transformador sklearn: textos -> matriz (n, 1) con el conteo de malsonantes."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, "tolist"):
            X = X.tolist()
        return np.asarray([[self.count_vulgar(str(t))] for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)

    @staticmethod
    def count_vulgar(text: str) -> float:
        return float(sum(1 for w in _prepare(text) if w in VULGAR_LEXICON))


if __name__ == "__main__":
    fe = VulgarityFeature()
    demos = [
        "que puto mal escribo, esto es una mierda",   # 2
        "joderrr el putooo hueso me duele",            # 2 (elongaciones)
        "me suelta la puta subnormal",                  # 2
        "que tengas un buen dia, un saludo",            # 0
    ]
    for d in demos:
        print(fe.transform([d])[0][0], "<<", d)
