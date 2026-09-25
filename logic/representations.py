"""Representaciones del texto para las Fases 2-4: texto limpio para BoW/TF-IDF y caracteristicas lexicas."""
import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from logic.feature_extraction import FeatureExtraction, NEW_FEATURE_NAMES
from logic.text_processing import TextProcessing

# Orden exacto en que get_features_lexical() devuelve su vector
ORIGINAL_NAMES = [
    'weighted_position', 'weighted_normalized',
    'label_mention', 'label_url', 'label_hashtag', 'label_emoji', 'label_retweets',
    'lexical_diversity', 'label_word',
    'first_person_singular', 'second_person_singular', 'third_person_singular',
    'first_person_plurar', 'second_person_plurar', 'third_person_plurar',
    'avg_word', 'kur_word', 'skew_word',
    'adverb_neg', 'adverb_time', 'adverb_place', 'adverb_mode', 'adverb_cant', 'adverb_all',
    'adjetives_neg', 'adjetives_pos', 'who_general', 'who_male', 'who_female',
]
ALL_NAMES = ORIGINAL_NAMES + NEW_FEATURE_NAMES

# Autor de cada caracteristica nueva (para la ablacion y la bitacora)
AUTHORS = {
    'Sebastian Sarmiento': ['emoji_count', 'emoji_pol', 'laugh_count', 'laugh_len', 'lex_pol_neg'],
    'Zuly Gonzalez': ['doubt_count', 'regionalism_count', 'intensifier_count'],
    'Daniel Contreras': ['social_pos', 'groserias'],
}

URL_RE = re.compile(r'(?i)(?:https?://|www\.)\S+')
EMOJI_RE = re.compile('[\U0001F000-\U000E007F☀-➿]')


def clean_text(text):
    """Preprocesamiento oficial del curso, entrada de BoW/TF-IDF."""
    return TextProcessing.transformer(str(text)) or ''


def lexical_text(text):
    """Como transformer(), pero conserva tildes y deja url, mention, hashtag y emoji como palabras.

    Es la variante del notebook del taller: con transformer() las etiquetas salen como [MENTION]
    y los emojis se borran, asi que get_features_lexical() nunca las encuentra.
    """
    t = unicodedata.normalize('NFC', str(text)).lower()
    t = URL_RE.sub(' url ', t)
    t = re.sub(r'@\w{1,40}', ' mention ', t)
    t = re.sub(r'#\w{1,40}', ' hashtag ', t)
    t = EMOJI_RE.sub(' emoji ', t)
    t = TextProcessing.remove_patterns(t)
    return re.sub(r'\s+', ' ', t).strip()


class LexicalFeatures(BaseEstimator, TransformerMixin):
    """Adaptador scikit-learn: tweets crudos -> matriz de caracteristicas lexicas.

    No aprende nada de los datos (fit no mira X), por eso puede calcularse una sola vez sin fuga.
    `columns` elige que caracteristicas devolver (None = todas, en el orden de ALL_NAMES).
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.fe_ = FeatureExtraction('es')
        return self

    def transform(self, X):
        if not hasattr(self, 'fe_'):
            self.fit(X)
        rows = []
        for text in X:
            text = str(text)
            original = self.fe_.get_features_lexical(lexical_text(text))
            if original is None:
                original = np.zeros(len(ORIGINAL_NAMES))
            rows.append(np.concatenate([original, self.fe_.get_features_nuevas(text)]))
        matrix = np.nan_to_num(np.asarray(rows, dtype=np.float64))
        index = [ALL_NAMES.index(c) for c in self.get_feature_names_out()]
        return matrix[:, index]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(ALL_NAMES if self.columns is None else list(self.columns), dtype=object)
