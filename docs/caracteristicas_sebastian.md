# Características léxicas — Sebastián Sarmiento

Implementación: `FeatureExtraction.get_features_nuevas()` en `logic/feature_extraction.py`.
Listas en `logic/lexical_features.py` (`emoji_pos`, `emoji_neg`, `laugh`, `negation`, `adversative`).
Pruebas: `tests/test_nuevas_features.py` (13 pruebas con tweets reales del train).

Se calculan sobre el **texto crudo**: `transformer()` aplica `proper_encoding()` (NFD + ASCII), que
borra los emojis y las tildes antes de que se puedan contar.

| ID | Columnas | Hipótesis lingüística | Cálculo |
|---|---|---|---|
| `emoji_pol` | `emoji_count`, `emoji_pol` | El emoji es la marca afectiva explícita del tweet; su polaridad separa P de N y BoW no la ve | Regex de rangos Unicode de emoji; +1 / −1 según el signo del Emoji Sentiment Ranking (Kralj Novak et al., 2015) |
| `laugh_count` | `laugh_count`, `laugh_len` | La risa marca tono informal o humorístico; la longitud mide la intensidad (`jajajaja` > `jaja`) | Regex `(j[aeiou]){2,}j?` + lista cerrada (`xd`, `lol`, `lmao`, `jsjs`, 🤣) |
| `lex_pol_neg` | `lex_pol_neg` | "no es bueno" es negativo aunque `bueno` sea positivo; BoW cuenta `bueno` igual en ambos casos | ML-SentiCon. La negación invierte el signo hasta la puntuación o una adversativa (`pero`, `aunque`, `sino`). Suma / nº de palabras con polaridad |

## Verificación en el train (1008 tweets)

| | N | NEU | NONE | P | Tweets con valor ≠ 0 |
|---|---|---|---|---|---|
| `emoji_pol` (media) | 0.012 | 0.023 | 0.014 | **0.079** | 2.9% |
| `laugh_len` (media) | 0.364 | 0.195 | 0.065 | 0.264 | 3.7% |
| `lex_pol_neg` (media) | **−0.044** | 0.054 | 0.053 | **0.159** | 61.9% |

- `lex_pol_neg` es la que más separa: negativa en N y la más alta en P.
- `emoji_pol` va en la dirección esperada (máxima en P), pero solo el 2.9% de los tweets tiene emojis.
- `laugh_len` no separa por polaridad: la risa aparece también en tweets negativos e irónicos
  (*"Ayer me quedé dormido xdddd lo peor es que sigo reventado"*, N).

## Limitaciones

- ML-SentiCon es un léxico de **lemas**: formas conjugadas como `gusta` no están, por eso
  *"no me gusta"* no se captura.
- `lol` también es el nombre de un videojuego (*"jugar más partidas al lol"*) → falso positivo de risa.
- La polaridad del emoji es solo el signo (+1/−1), no el puntaje continuo del ranking.
