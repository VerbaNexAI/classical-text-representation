"""
IvanCruz_JuanOchoa_features.py
==============================
Características léxicas adicionales para clasificación de polaridad en tweets
en español (TASS 2018).

La no redundancia se documenta en
``docs/caracteristicas_IvanCruz_JuanOchoa.md``.

Diseño
------
* ``FEATURE_SPECS`` declara cada característica con su grupo (para la ablación)
  y su hipótesis lingüística.
* ``extract_features(texto)`` devuelve un dict {nombre: valor} para UN tweet.
* ``LexicalFeaturizer`` es el adaptador scikit-learn (fit/transform), sin
  estado aprendido, compatible con Pipeline y con selección por grupo.
* ``audit_features`` verifica que las características no sean constantes ni
  proxies de longitud (misma función que usó la Fase 0).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# 1. Especificación de características (documentación ejecutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    hypothesis: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # --- Expresividad -----------------------------------------------------
    FeatureSpec(
        "elongated_count", "expresividad",
        "Alargar letras ('buenooo', 'noooo') es un sustituto escrito de la "
        "prosodia: marca énfasis emocional y suele acompañar a polaridad fuerte "
        "(P o N). Se ignora en siglas ('CCC') y en números.",
    ),
    FeatureSpec(
        "upper_ratio", "expresividad",
        "Palabras completas en mayúsculas ('NO ME GUSTA') equivalen a gritar. "
        "Se normaliza por el número de palabras elegibles para no depender de "
        "la longitud del tweet.",
    ),
    FeatureSpec(
        "exclam_run_max", "expresividad",
        "Repetir '!' ('¡¡¡bien!!!') amplifica la emoción; la longitud de la "
        "racha máxima captura intensidad, no solo presencia.",
    ),
    FeatureSpec(
        "question_run_max", "expresividad",
        "Repetir '?' ('¿¿qué??') suele indicar incredulidad, molestia o duda "
        "retórica, más frecuente en N y NEU que en P.",
    ),
    FeatureSpec(
        "interrobang_count", "expresividad",
        "La combinación '?!' / '!?' expresa sorpresa o indignación mezcladas; "
        "es una señal que la simple cuenta de '!' y '?' no distingue.",
    ),
    # --- Modificadores ----------------------------------------------------
    FeatureSpec(
        "intensifier_count", "modificadores",
        "Intensificadores ('muy', 'súper', 'demasiado', 'totalmente') refuerzan "
        "la polaridad del adjetivo o verbo que modifican.",
    ),
    FeatureSpec(
        "attenuator_count", "modificadores",
        "Atenuadores ('algo', 'un poco', 'apenas', 'casi') suavizan la "
        "valoración y son más comunes en opiniones matizadas o neutras.",
    ),
    # --- Coloquial --------------------------------------------------------
    FeatureSpec(
        "interjection_count", "coloquial",
        "Interjecciones ('ay', 'uf', 'wow', 'ojalá') señalan reacción "
        "emocional inmediata; pertenecen al registro afectivo y suelen "
        "acompañar opiniones polarizadas.",
    ),
    FeatureSpec(
        "regionalism_count", "coloquial",
        "Marcadores dialectales ('wey', 'chido', 'mola', 'guay', 'chevere') "
        "capturan variedades del español con su propio registro evaluativo "
        "(p. ej. 'chido', 'mola' = positivo).",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {}
for _s in FEATURE_SPECS:
    FEATURE_GROUPS[_s.group] = FEATURE_GROUPS.get(_s.group, ()) + (_s.name,)


# ---------------------------------------------------------------------------
# 2. Normalización y tokenización
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_TOKEN_RE = re.compile(r"[a-zñü0-9_]+|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"^[a-zñü0-9_]+$")
_PLACEHOLDER = "\ue000"  # para proteger la ñ al quitar tildes


def _strip_accents(text: str) -> str:
    """Minúsculas y sin tildes, conservando la ñ."""
    text = text.lower().replace("ñ", _PLACEHOLDER)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text.replace(_PLACEHOLDER, "ñ")


def _canon(token: str) -> str:
    """Colapsa toda racha de letras repetidas a una sola ('buenooo' -> 'bueno')."""
    return re.sub(r"([a-zñü])\1+", r"\1", token)


def clean_text(text: object) -> str:
    """Quita URLs y menciones. Conserva '#' y la palabra del hashtag."""
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    return _MENTION_RE.sub(" ", text)


def _is_word(tok: str) -> bool:
    return bool(_WORD_RE.match(tok))


# ---------------------------------------------------------------------------
# 3. Listas de palabras (semillas auditadas sobre train)
# ---------------------------------------------------------------------------

_INTENSIFIERS = [
    "muy", "super", "mega", "hiper", "ultra", "tan", "demasiado", "demasiada",
    "demasiados", "demasiadas", "tremendo", "tremenda", "tremendamente",
    "sumamente", "increiblemente", "realmente", "totalmente", "absolutamente",
    "completamente", "extremadamente", "altamente",
]
# 'bien' se excluye a propósito porque ya lo cubre el léxico de polaridad del taller.

_ATTENUATORS = [
    "algo", "poco", "poca", "pocos", "pocas", "casi", "apenas",
    "ligeramente", "relativamente", "un poco", "un tanto", "mas o menos",
    "algo asi",
]
# 'medio' se excluye por evidencia en train ('36 y medio', 'medio demonio').

_INTERJECTIONS = [
    "ay", "uy", "uf", "uff", "wow", "guau", "ojala", "ea", "ole", "bah",
    "puf", "hey", "eh", "oh", "ah", "uh", "ups", "oops", "caramba", "caray",
    "ufa", "epa", "hurra", "bravo", "zas", "jope",
]
# 'vaya' se excluye: en train aparece 3/6 veces como verbo 'ir' ('que os vaya bien').

_REGIONALISMS = [
    "wey", "guey", "chido", "chida", "pucha", "chuta", "guay", "chevere",
    "bacan", "boludo", "boluda", "pibe", "pibes", "vos", "mola", "ostras",
    "ahorita", "po", "cachai", "weon", "huevon",
]
# 'che' se excluye: en train solo aparece como "el Che" (Guevara).


# ---------------------------------------------------------------------------
# 4. Extractor
# ---------------------------------------------------------------------------

_RAW_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_]+")
_ELONG_RE = re.compile(r"([a-zñü])\1{2,}")
_EXCL_RUN_RE = re.compile(r"[!¡]+")
_QUEST_RUN_RE = re.compile(r"[?¿]+")
_INTERROBANG_RE = re.compile(r"(?:!\s*\?|\?\s*!)")
_UPPER_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")


def _phrase_regex(phrases: Sequence[str]) -> re.Pattern:
    """Regex sobre texto ya canónico y separado por espacios; frases largas primero."""
    canon = sorted({_canon(_strip_accents(p)) for p in phrases}, key=len, reverse=True)
    return re.compile(r"(?<!\S)(?:" + "|".join(re.escape(p) for p in canon) + r")(?!\S)")


class _Extractor:
    """Instancia sin estado; precalcula regexes de listas."""

    def __init__(self) -> None:
        self.intens_re = _phrase_regex(_INTENSIFIERS)
        self.atten_re = _phrase_regex(_ATTENUATORS)
        self.interj = {_canon(_strip_accents(w)) for w in _INTERJECTIONS}
        self.regio = {_canon(_strip_accents(w)) for w in _REGIONALISMS}

    def __call__(self, text: object) -> dict[str, float]:
        clean = clean_text(text)
        toks = _TOKEN_RE.findall(_strip_accents(clean))
        words = [t for t in toks if _is_word(t)]
        canon_words = [_canon(t) for t in words]
        canon_text = " " + " ".join(canon_words) + " "

        return {
            "elongated_count": float(self._count_elongated(clean)),
            "upper_ratio": float(self._upper_ratio(clean)),
            "exclam_run_max": float(max((len(m) for m in _EXCL_RUN_RE.findall(clean)), default=0)),
            "question_run_max": float(max((len(m) for m in _QUEST_RUN_RE.findall(clean)), default=0)),
            "interrobang_count": float(len(_INTERROBANG_RE.findall(clean))),
            "intensifier_count": float(len(self.intens_re.findall(canon_text))),
            "attenuator_count": float(len(self.atten_re.findall(canon_text))),
            "interjection_count": float(sum(w in self.interj for w in canon_words)),
            "regionalism_count": float(sum(w in self.regio for w in canon_words)),
        }

    def _count_elongated(self, clean: str) -> int:
        n = 0
        for t in _RAW_WORD_RE.findall(clean):
            t_norm = _strip_accents(t)
            if not _ELONG_RE.search(t_norm):
                continue
            if t.isdigit():
                continue
            if t.isupper() and len(t) <= 3:   # siglas: 'CCC', 'III'
                continue
            n += 1
        return n

    def _upper_ratio(self, clean: str) -> float:
        eligible = [
            w for w in _UPPER_WORD_RE.findall(clean)
            if len(w) >= 2 and w != "RT"
        ]
        if not eligible:
            return 0.0
        return sum(w.isupper() for w in eligible) / len(eligible)


_DEFAULT_EXTRACTOR = _Extractor()


def extract_features(text: object) -> dict[str, float]:
    """Características de UN tweet como dict {nombre: valor}."""
    return _DEFAULT_EXTRACTOR(text)


# ---------------------------------------------------------------------------
# 5. Adaptador scikit-learn
# ---------------------------------------------------------------------------

class LexicalFeaturizer(BaseEstimator, TransformerMixin):
    """
    Transformador scikit-learn: lista de textos -> matriz densa (n, k).

    Parameters
    ----------
    groups : sequence[str] | None
        Grupos de ``FEATURE_GROUPS`` a incluir (None = todos). Para ablación.
    features : sequence[str] | None
        Nombres concretos (tiene prioridad sobre ``groups``).

    Notas
    -----
    Transformador SIN ESTADO: ``fit`` no aprende nada de los datos, así que no
    puede filtrar información del test. El escalado debe ir en un paso
    posterior del Pipeline, ajustado solo en train.
    """

    def __init__(self, groups=None, features=None):
        self.groups = groups
        self.features = features

    def _select(self) -> list[str]:
        if self.features is not None:
            unknown = set(self.features) - set(FEATURE_NAMES)
            if unknown:
                raise ValueError(f"Características desconocidas: {sorted(unknown)}")
            return [n for n in FEATURE_NAMES if n in set(self.features)]
        if self.groups is not None:
            unknown = set(self.groups) - set(FEATURE_GROUPS)
            if unknown:
                raise ValueError(f"Grupos desconocidos: {sorted(unknown)}")
            wanted = {n for g in self.groups for n in FEATURE_GROUPS[g]}
            return [n for n in FEATURE_NAMES if n in wanted]
        return list(FEATURE_NAMES)

    def fit(self, X, y=None):
        self.feature_names_ = self._select()
        return self

    def transform(self, X):
        names = getattr(self, "feature_names_", None) or self._select()
        rows = [[extract_features(t)[n] for n in names] for t in X]
        return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            getattr(self, "feature_names_", None) or self._select(), dtype=object
        )


# ---------------------------------------------------------------------------
# 6. Auditoría (misma firma que la usada en la Fase 0)
# ---------------------------------------------------------------------------

def audit_features(
    texts: Sequence[str],
    extractor: Callable[[str], Mapping[str, float]] = extract_features,
    preprocess: Callable[[str], str] | None = None,
    length_corr_threshold: float = 0.7,
):
    """
    Devuelve un DataFrame con una fila por característica:
      * activation_rate : fracción de tweets con valor distinto de 0.
      * is_constant     : True si no varía en todo el corpus.
      * corr_n_tokens   : correlación de Pearson con el número de tokens.
      * length_proxy    : True si |corr| >= length_corr_threshold.
      * pct_changed_by_preprocess : % de tweets cuyo valor cambia al aplicar
        ``preprocess`` antes de extraer.
    """
    import pandas as pd

    raw = pd.DataFrame([dict(extractor(t)) for t in texts])
    n_tokens = np.array(
        [len(_TOKEN_RE.findall(_strip_accents(clean_text(t)))) for t in texts],
        dtype=float,
    )

    processed = None
    if preprocess is not None:
        processed = pd.DataFrame([dict(extractor(preprocess(t))) for t in texts])

    out = []
    for col in raw.columns:
        v = raw[col].to_numpy(dtype=float)
        const = bool(np.nanstd(v) == 0)
        corr = (
            float("nan") if const or np.std(n_tokens) == 0
            else float(np.corrcoef(v, n_tokens)[0, 1])
        )
        row = {
            "feature": col,
            "activation_rate": float(np.mean(v != 0)),
            "n_unique": int(len(np.unique(v))),
            "std": float(np.nanstd(v)),
            "is_constant": const,
            "corr_n_tokens": corr,
            "length_proxy": bool(abs(corr) >= length_corr_threshold) if not np.isnan(corr) else False,
        }
        if processed is not None and col in processed.columns:
            row["pct_changed_by_preprocess"] = float(
                100 * np.mean(~np.isclose(v, processed[col].to_numpy(dtype=float)))
            )
        out.append(row)
    return pd.DataFrame(out).set_index("feature")