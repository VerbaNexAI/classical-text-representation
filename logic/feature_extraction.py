import re
import sys
from nltk import TweetTokenizer
from scipy.stats import kurtosis, skew
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from logic.text_processing import TextProcessing
from logic.utils import Utils
from logic.lexical_features import lexical_es, lexical_en, polaridad_es, negacion_es, fin_alcance_negacion_es


class FeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, lang='es'):
        try:
            self.lexical = lexical_es if lang == 'es' else lexical_en
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error FeatureExtraction: {0}'.format(e))

    def fit(self, x, y=None):
        return self

    def transform(self, list_messages):
        try:
            result = self.get_features(list_messages)
            return result
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error transform: {0}'.format(e))

    def get_features(self, messages: str):
        try:

            features = list(abs(self.get_features_lexical(messages)))
            result = np.array(features, dtype=np.float32)
            return result
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_features: {0}'.format(e))
            return None

    def get_features_lexical(self, message):
        result = None
        try:
            lexical = self.lexical
            text_tokenizer = TweetTokenizer()
            tags = ('mention', 'url', 'hashtag', 'emoji', 'rt')
            vector = dict()
            tokens_text = text_tokenizer.tokenize(message)
            if len(tokens_text) > 0:
                vector['weighted_position'], vector['weighted_normalized'] = self.weighted_position(tokens_text)

                vector['label_mention'] = float(sum(1 for word in tokens_text if word == 'mention'))
                vector['label_url'] = float(sum(1 for word in tokens_text if word == 'url'))
                vector['label_hashtag'] = float(sum(1 for word in tokens_text if word == 'hashtag'))
                vector['label_emoji'] = float(sum(1 for word in tokens_text if word == 'emoji'))
                vector['label_retweets'] = float(sum(1 for word in tokens_text if word == 'rt'))

                vector['lexical_diversity'] = self.lexical_diversity(message)

                label_word = vector['label_mention'] + vector['label_url'] + vector['label_hashtag']
                label_word = label_word + vector['label_emoji'] + vector['label_retweets']
                vector['label_word'] = float(len(tokens_text) - label_word)

                vector['first_person_singular'] = float(
                    sum(1 for word in tokens_text if word in lexical['first_person_singular']))
                vector['second_person_singular'] = float(
                    sum(1 for word in tokens_text if word in lexical['second_person_singular']))
                vector['third_person_singular'] = float(
                    sum(1 for word in tokens_text if word in lexical['third_person_singular']))
                vector['first_person_plurar'] = float(
                    sum(1 for word in tokens_text if word in lexical['first_person_plurar']))
                vector['second_person_plurar'] = float(
                    sum(1 for word in tokens_text if word in lexical['second_person_plurar']))
                vector['third_person_plurar'] = float(
                    sum(1 for word in tokens_text if word in lexical['third_person_plurar']))

                vector['avg_word'] = np.nanmean([len(word) for word in tokens_text if word not in tags])
                vector['avg_word'] = vector['avg_word'] if not np.isnan(vector['avg_word']) else 0.0
                vector['avg_word'] = round(vector['avg_word'], 4)

                vector['kur_word'] = kurtosis([len(word) for word in tokens_text if word not in tags])
                vector['kur_word'] = vector['kur_word'] if not np.isnan(vector['kur_word']) else 0.0
                vector['kur_word'] = round(vector['kur_word'], 4)

                vector['skew_word'] = skew(np.array([len(word) for word in tokens_text if word not in tags]))
                vector['skew_word'] = vector['skew_word'] if not np.isnan(vector['skew_word']) else 0.0
                vector['skew_word'] = round(vector['skew_word'], 4)

                # adverbios
                vector['adverb_neg'] = sum(1 for word in tokens_text if word in lexical['adverb_neg'])
                vector['adverb_neg'] = float(vector['adverb_neg'])

                vector['adverb_time'] = sum(1 for word in tokens_text if word in lexical['adverb_time'])
                vector['adverb_time'] = float(vector['adverb_time'])

                vector['adverb_place'] = sum(1 for word in tokens_text if word in lexical['adverb_place'])
                vector['adverb_place'] = float(vector['adverb_place'])

                vector['adverb_mode'] = sum(1 for word in tokens_text if word in lexical['adverb_mode'])
                vector['adverb_mode'] = float(vector['adverb_mode'])

                vector['adverb_cant'] = sum(1 for word in tokens_text if word in lexical['adverb_cant'])
                vector['adverb_cant'] = float(vector['adverb_cant'])

                vector['adverb_all'] = float(vector['adverb_neg'] + vector['adverb_time'] + vector['adverb_place'])
                vector['adverb_all'] = float(vector['adverb_all'] + vector['adverb_mode'] + vector['adverb_cant'])

                vector['adjetives_neg'] = sum(1 for word in tokens_text if word in lexical['adjetives_neg'])
                vector['adjetives_neg'] = float(vector['adjetives_neg'])

                vector['adjetives_pos'] = sum(1 for word in tokens_text if word in lexical['adjetives_pos'])
                vector['adjetives_pos'] = float(vector['adjetives_pos'])

                vector['who_general'] = sum(1 for word in tokens_text if word in lexical['who_general'])
                vector['who_general'] = float(vector['who_general'])

                vector['who_male'] = sum(1 for word in tokens_text if word in lexical['who_male'])
                vector['who_male'] = float(vector['who_male'])

                vector['who_female'] = sum(1 for word in tokens_text if word in lexical['who_female'])
                vector['who_female'] = float(vector['who_female'])

                result = np.array(list(vector.values()))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_lexical_features: {0}'.format(e))
        return result

    @staticmethod
    def lexical_diversity(text):
        result = None
        try:
            text_out = re.sub(r"[\U00010000-\U0010ffff]", '', text)
            text_out = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
                r'|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                '', text_out)
            text_out = text_out.lower()
            result = round((len(set(text_out)) / len(text_out)), 4)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error lexical_diversity: {0}'.format(e))
        return result

    @staticmethod
    def weighted_position(tokens_text):
        result = None
        try:
            size = len(tokens_text)
            weighted_words = 0.0
            weighted_normalized = 0.0
            for w in tokens_text:
                weighted_words += 1 / (1 + tokens_text.index(w))
                weighted_normalized += (1 + tokens_text.index(w)) / size
            result = (weighted_words, weighted_normalized)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error weighted_position: {0}'.format(e))
        return result

    # ------------------------------------------------------------------
    # Fase 1 - Nuevas caracteristicas lexicas
    # ------------------------------------------------------------------

    @staticmethod
    def lexical_diversity_mattr(tokens_text, window=10):
        """
        Hipotesis linguistica: la diversidad lexica (TTR) cruda depende
        matematicamente de la longitud del texto (ver auditoria Fase 0).
        MATTR (Moving-Average Type-Token Ratio) corrige ese sesgo promediando
        el TTR calculado sobre ventanas moviles de tamano fijo, por lo que
        mide diversidad de vocabulario de forma comparable entre tweets
        cortos y largos. Reemplaza a 'lexical_diversity' (que ademas media
        diversidad de CARACTERES, no de palabras, por un error de
        implementacion detectado en la auditoria).
        """
        try:
            palabras = [w for w in tokens_text if w not in ('mention', 'url', 'hashtag', 'emoji', 'rt')]
            n = len(palabras)
            if n == 0:
                return 0.0
            if n <= window:
                return round(len(set(palabras)) / n, 4)
            ratios = []
            for i in range(0, n - window + 1):
                ventana = palabras[i:i + window]
                ratios.append(len(set(ventana)) / window)
            return round(float(np.mean(ratios)), 4)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error lexical_diversity_mattr: {0}'.format(e))
            return 0.0

    @staticmethod
    def elongated_words_count(raw_text):
        """
        Hipotesis linguistica: la elongacion de caracteres ('buenooo',
        'nooo') es un marcador ortografico de intensidad emocional tipico
        del espanol informal de Twitter, ausente en el registro formal que
        domina BoW/TF-IDF. Se calcula sobre el texto CRUDO (antes de
        transformer()), porque remove_patterns() no altera letras repetidas
        pero el resultado debe reflejar la ortografia original del tweet,
        no tokens ya normalizados.
        """
        try:
            texto = raw_text.lower()
            coincidencias = re.findall(r'([a-záéíóúüñ])\1{2,}', texto)
            return float(len(coincidencias))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error elongated_words_count: {0}'.format(e))
            return 0.0

    @staticmethod
    def punct_repetition(raw_text):
        """
        Hipotesis linguistica: la repeticion de signos de exclamacion o
        interrogacion ('genial!!!', 'en serio???') intensifica la carga
        emocional de un enunciado, una senal que BoW no puede capturar
        porque trata cada '!' como un token identico e independiente del
        contexto. Se calcula sobre el texto CRUDO porque remove_patterns()
        elimina toda la puntuacion antes de tokenizar.
        """
        try:
            grupos = re.findall(r'[!¡]{2,}|[?¿]{2,}', raw_text)
            return float(len(grupos))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error punct_repetition: {0}'.format(e))
            return 0.0

    @staticmethod
    def lex_pol_neg(tokens_text):
        """
        Hipotesis linguistica: un conteo simple de palabras positivas y
        negativas (como ya hacen 'adjetives_pos'/'adjetives_neg') falla
        cuando la palabra esta bajo el alcance de una negacion ('no me
        gusta' cuenta 'gusta' como positivo, siendo el enunciado negativo).
        Esta caracteristica invierte la polaridad de las palabras del
        lexico que caen dentro del alcance de un marcador de negacion,
        hasta el siguiente limite (conjuncion o fin del tweet). No duplica
        'adjetives_pos/neg': esas cuentan ocurrencias crudas, esta calcula
        un puntaje neto con la negacion resuelta.
        """
        try:
            score = 0.0
            en_negacion = False
            for w in tokens_text:
                if w in fin_alcance_negacion_es:
                    en_negacion = False
                    continue
                if w in negacion_es:
                    en_negacion = True
                    continue
                signo = 0
                if w in polaridad_es['positivo']:
                    signo = 1
                elif w in polaridad_es['negativo']:
                    signo = -1
                if signo != 0 and en_negacion:
                    signo = -signo
                score += signo
            return float(score)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error lex_pol_neg: {0}'.format(e))
            return 0.0
