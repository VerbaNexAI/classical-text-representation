"""Característica léxica: diminutivos afectivos (TASS 2018).

Una sola característica, `diminutive_count`: número de diminutivos afectivos en el
tuit ('besito', 'ratito', 'pobrecito', 'poquito'...).

Hipótesis lingüística
---------------------
El diminutivo en español no siempre reduce el tamaño: con frecuencia expresa
**afecto, cercanía o cortesía** ('un ratito', 'pobrecito', 'despacito'). Es un
marcador de subjetividad afectiva que **escasea en los tuits sin carga (NONE)** y
tiende a acompañar polaridad **positiva**. La Bolsa de Palabras ve 'casa' y 'casita'
como tokens distintos y no captura ese matiz común.

Detección (precisión + cobertura, para que "no se escape" nada):
  1. **Lista curada** de diminutivos afectivos frecuentes en tuits.
  2. **Regla morfológica** para los sufijos productivos -it[o/a](s) y -ill[o/a](s).
  3. **Lista de exclusión** de falsos positivos frecuentes: verbos ('necesito',
     'repito'), adjetivos/sustantivos que terminan igual pero no son diminutivos
     ('bonito', 'infinito', 'pastilla', 'semilla', 'maravilla') y topónimos
     ('Sevilla'). Además se eliminan menciones (@), hashtags (#) y URLs para no
     contar nombres de usuario.

Se implementa como un transformador compatible con scikit-learn (`fit`/`transform`),
*stateless*, para unirlo con BoW/TF-IDF mediante `FeatureUnion` o `hstack`.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = ["diminutive_count"]

# 1. Lista curada de diminutivos afectivos (minúscula, sin acentos).
DIMINUTIVE_LIST = {
    "besito", "besitos", "ratito", "ratitos", "ratico", "poquito", "poquita",
    "poquitos", "poquitico", "carinito", "gatito", "gatita", "perrito", "perrita",
    "chiquito", "chiquita", "solito", "solita", "cerquita", "prontito", "calentito",
    "calentita", "pobrecito", "pobrecita", "casita", "cosita", "cositas", "carita",
    "despacito", "hermanito", "hermanita", "amiguito", "amiguita", "momentito",
    "pequenito", "pequenita", "tranquilito", "calladito", "pesadito", "clarito",
    "fresquito", "fresquita", "completito", "horita", "horitas", "pajarito",
    "palomitas", "corazoncito", "cansadito", "todito", "ahorita", "trocito",
    "cortito", "abrazito", "florecita", "manita", "cafecito", "chiquitin",
}

# 2. Regla morfológica: sufijos diminutivos productivos.
_DIM_SUFFIX = re.compile(r"^[a-z]{3,}(?:it|ill)[oa]s?$")

# 3. Exclusiones: palabras que terminan en esos sufijos pero NO son diminutivos.
DIMINUTIVE_EXCLUDE = {
    # verbos y sustantivos en -ito/-ita
    "necesito", "necesita", "necesitas", "invito", "repito", "visito", "visita",
    "visitas", "cita", "citas", "deposito", "medito", "imito", "evito", "grito",
    "gritos", "admito", "credito", "debito", "requisito", "proposito", "transito",
    "exito", "exitos", "habito", "habitos", "limito", "derrito", "escrito",
    "escritos", "bonito", "bonita", "bonitos", "bonitas", "maldito", "maldita",
    "malditos", "infinito", "infinita", "favorito", "favorita", "favoritos",
    "favoritas", "exquisito", "gratuito", "hipocrita", "mosquito", "mosquitos",
    "delito", "delitos", "apetito", "circuito", "lolita", "monysito", "imunitas",
    # palabras en -illo/-illa que no son diminutivos
    "sevilla", "pastilla", "pastillas", "semilla", "semillas", "silla", "sillas",
    "orilla", "orillas", "mejilla", "mejillas", "maravilla", "maravillas", "rodilla",
    "rodillas", "tobillo", "tobillos", "carrillo", "tortilla", "tortillas", "ardilla",
    "pesadilla", "pesadillas", "cosquillas", "mantequilla", "zapatilla", "zapatillas",
    "camilla", "costilla", "costillas", "vainilla", "taquilla", "anillo", "anillos",
    "cepillo", "ladrillo", "bolsillo", "bolsillos", "cigarrillo", "martillo", "pillo",
    "brillo", "amarillo", "amarilla", "sencillo", "sencilla", "cuchillo", "bombilla",
    "capilla", "villa", "pandilla", "rejilla", "plantilla", "manzanilla", "barbilla",
    "morcilla", "patilla", "natillas", "comillas", "comilla", "olmillo",
}

_STRIP_RE = re.compile(r"@\w+|#\w+|https?://\S+|www\.\S+")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

FEATURE_DOC = {
    "diminutive_count":
        "Número de diminutivos afectivos en el tuit. Hipótesis: afecto/cercanía; "
        "escasea en NONE y tiende a P.",
}


def _prepare(text: str) -> list:
    """Quita menciones/hashtags/URLs y pasa a minúscula sin acentos."""
    text = _STRIP_RE.sub(" ", text)
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _WORD_RE.findall(text)


class DiminutiveFeature(BaseEstimator, TransformerMixin):
    """Transformador sklearn: textos -> matriz (n, 1) con el conteo de diminutivos."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, "tolist"):
            X = X.tolist()
        return np.asarray([[self.count_diminutives(str(t))] for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)

    @staticmethod
    def count_diminutives(text: str) -> float:
        c = 0
        for w in _prepare(text):
            if w in DIMINUTIVE_LIST:
                c += 1
            elif _DIM_SUFFIX.match(w) and w not in DIMINUTIVE_EXCLUDE:
                c += 1
        return float(c)


if __name__ == "__main__":
    fe = DiminutiveFeature()
    demos = [
        "un ratito bonito, besitos",                 # besito(s) + ratito (bonito excluido) = 2
        "me dice un pajarito y me quedo calladito",  # 2
        "necesito una pastilla en Sevilla",           # 0 (excluidos)
        "el partido es a las tres",                   # 0
    ]
    for d in demos:
        print(fe.transform([d])[0][0], "<<", d)
