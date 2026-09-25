# Características léxicas — Fabián Rincón y Jhon Rodríguez

**Implementación:** `FeatureExtraction.get_features_lexical()` en `logic/feature_extraction.py`.

Las características se integran al modelo mediante `LexicalFeaturesTransformer`, que obtiene las características léxicas a partir de `FeatureExtraction` y las combina con la representación TF-IDF mediante `FeatureUnion`.

Dentro del conjunto de características léxicas se añadieron seis nuevos grupos de vocabulario: `reaction_words`, `support_words`, `apology_words`, `recommendation_words`, `preference_words` y `experience_words`.

| ID                     | Columna                | Hipótesis lingüística                                                                                                                              | Cálculo                                                                                                                                            |
| ---------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reaction_words`       | `reaction_words`       | Las palabras de reacción espontánea pueden expresar una respuesta emocional o una valoración positiva frente a una situación.                      | palabras del vocabulario de reacción presentes en el texto: `wow`, `increíble`, `brutal`, `impresionante`, `sorprendente`, `impactante`. |
| `support_words`        | `support_words`        | Las expresiones de apoyo o ánimo pueden reflejar una actitud positiva hacia una persona, situación o acontecimiento.                               | palabras del vocabulario de apoyo: `apoyo`, `respaldo`, `fuerza`, `ánimo`, `adelante`, `vamos`.                                          |
| `apology_words`        | `apology_words`        | Las expresiones de disculpa o arrepentimiento pueden estar asociadas con situaciones de carácter negativo o con una valoración desfavorable.       | palabras del vocabulario de disculpas: `perdón`, `disculpa`, `disculpas`, `perdóname`.                                                   |
| `recommendation_words` | `recommendation_words` | Las expresiones de recomendación pueden indicar una valoración positiva o una intención de sugerir una opción o experiencia.                       | palabras del vocabulario de recomendación: `recomiendo`, `recomendado`, `sugiero`, `aconsejo`.                                           |
| `preference_words`     | `preference_words`     | Las expresiones de preferencia permiten identificar opiniones o valoraciones personales sobre una opción determinada.                              | palabras del vocabulario de preferencias: `prefiero`, `elegiría`, `favorito`, `favorita`, `preferido`, `preferida`.                      |
| `experience_words`     | `experience_words`     | Las expresiones relacionadas con experiencias personales pueden aportar información sobre situaciones vividas directamente por el autor del tweet. | palabras del vocabulario de experiencias: `probé`, `viví`, `experimenté`, `conocí`, `utilicé`, `visité`.                                 |

## Verificación en el conjunto de entrenamiento

Las seis características nuevas se incorporaron al conjunto de características léxicas utilizado por el modelo. Para evaluar su aporte, se realizó una **ablación**, retirando los nuevos grupos de características y comparando el comportamiento del modelo con la representación completa.

La ablación se realizó en dos grupos de tres características:

| Grupo       | Características                                                |
| ----------- | -------------------------------------------------------------- |
| **Grupo 1** | `reaction_words`, `support_words`, `apology_words`             |
| **Grupo 2** | `recommendation_words`, `preference_words`, `experience_words` |

El objetivo de esta comparación es determinar si la incorporación de estos vocabularios aporta información adicional a la representación TF-IDF y a las demás características léxicas utilizadas por `FeatureExtraction`.

## Integración con TF-IDF

Las características léxicas no reemplazan la representación basada en texto. El modelo utiliza una combinación de ambas representaciones:

**TF-IDF + características léxicas de `FeatureExtraction`**

La representación TF-IDF captura patrones de palabras y caracteres presentes en los tweets, mientras que las características léxicas aportan información específica basada en categorías lingüísticas definidas previamente.

En particular, las seis características nuevas buscan incorporar información que puede no estar representada de manera explícita por la frecuencia de los términos en el modelo TF-IDF.

## Limitaciones

* Los vocabularios utilizados son **listas cerradas de palabras**, por lo que únicamente se detectan las expresiones incluidas explícitamente en cada categoría.
* Las variaciones ortográficas, errores de escritura, sinónimos o expresiones equivalentes que no estén incluidas en los vocabularios no son detectadas.
* La presencia de una palabra no garantiza por sí sola una determinada polaridad. Por ejemplo, una expresión de apoyo puede aparecer dentro de un tweet con contexto negativo o irónico.
* Las características se basan principalmente en la **presencia o frecuencia de palabras específicas**, por lo que no capturan completamente el contexto en el que se utilizan.
* Las características nuevas se evalúan mediante ablación en grupos de tres, por lo que este análisis permite estudiar el aporte de cada grupo, pero no el efecto aislado de cada palabra individual.
