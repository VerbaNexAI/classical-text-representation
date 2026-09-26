# Características léxicas — Juan David Tejedor Medina y Miguel Gerardo Moreno

Este archivo describe las trece columnas que añadimos al extractor de TASS 2018. Diez recogen formas de expresión frecuentes en tuits y tres amplían las señales de negación, polaridad y puntuación. Se calculan sobre el texto original, antes de pasar a minúsculas, quitar signos o limpiar emojis.

## Hipótesis y cálculo

| Característica | Qué esperamos que aporte | Cómo se calcula |
|---|---|---|
| `elongated_word_count` | Alargar una palabra puede marcar emoción o énfasis. | Número de palabras con tres o más letras iguales seguidas; se excluyen las risas. |
| `elongation_excess` | El grado de alargamiento puede distinguir un énfasis leve de uno fuerte. | Letras repetidas que sobran a partir de la segunda en esas palabras. |
| `uppercase_word_count` | Una palabra en mayúsculas puede expresar sorpresa, entusiasmo o enfado. | Número de palabras alfabéticas de al menos dos letras escritas completamente en mayúsculas. |
| `uppercase_word_ratio` | La proporción evita que el simple tamaño del tuit determine el conteo. | Palabras en mayúsculas divididas entre las palabras alfabéticas del tuit. |
| `exclamation_run_max` | Varias exclamaciones seguidas pueden intensificar una valoración. | Longitud de la racha más larga de signos `!` o `¡`. |
| `question_run_max` | Varias interrogaciones seguidas pueden acompañar duda, sorpresa o reclamo. | Longitud de la racha más larga de signos `?` o `¿`. |
| `intensifier_count` | Palabras como «muy» o «demasiado» refuerzan una opinión. | Conteo de palabras en la lista de intensificadores incluida en el cuadernillo. |
| `attenuator_count` | Palabras como «poco» o «quizás» pueden suavizar una valoración. | Conteo de palabras en la lista de atenuadores. |
| `interjection_count` | Una interjección puede indicar una reacción inmediata. | Conteo de interjecciones reconocidas por la lista léxica. |
| `colloquial_count` | Algunas expresiones informales aportan pistas sobre el tono. | Conteo de palabras presentes en la lista de coloquialismos. |
| `negated_positive_count` | Negar una palabra positiva puede cambiar el sentido del mensaje. | Cuenta términos positivos de ML-SentiCon después de una negación activa, hasta puntuación o una adversativa; la expresión «no solo» se exceptúa. |
| `positive_negative_ratio` | El balance de palabras valorativas puede ayudar a distinguir P de N. | `(positivas + 1) / (negativas + 1)`, con suavizado para evitar una división entre cero. |
| `punctuation_repeat_excess` | Repetir signos puede marcar énfasis adicional. | Suma los signos sobrantes después del primero en cada racha de exclamación o interrogación. |

## Comprobaciones con los datos

En los 1.008 tuits de entrenamiento, las diez columnas expresivas no resultaron constantes. Por ejemplo, `exclamation_run_max` se activó en 125 tuits, `intensifier_count` en 128, `question_run_max` en 78 y `elongated_word_count` en 29. De las tres columnas complementarias, `negated_positive_count` tomó un valor distinto de cero en 59 tuits y `punctuation_repeat_excess` en 49. `positive_negative_ratio` tiene un valor de base igual a 1 cuando no hay palabras del léxico; por eso su valor distinto de cero en todos los tuits **no significa** que todos contengan opinión.

Las pruebas del módulo usan tuits reales de entrenamiento. En el tuit `768231706746912769`, «siiii» produce una palabra alargada y un exceso de dos letras. En el `768212591105703936`, «seguro!!» deja una racha máxima de dos exclamaciones y un signo repetido de exceso. En el `768055614413086720`, la prueba reconoce dos intensificadores y un atenuador. También hay pruebas específicas para las tres columnas complementarias. La ejecución guardada en el cuadernillo del módulo muestra **22 pruebas aprobadas** y un vector final de **43 columnas**.

En la comparación del cuadernillo de clasificación, la unión Word2Vec + léxico obtuvo F1 macro **0,4428** en validación cruzada con SVM lineal. Al quitar los rasgos nuevos y dejar solo los históricos, el F1 fue **0,4228**. Esta diferencia es descriptiva del experimento: los rasgos nuevos incluyen los cinco aportados previamente y los trece documentados aquí; no permite asignar toda la mejora a una sola columna. En los 506 tuits reservados, el modelo elegido alcanzó F1 macro **0,4187**.

## Relación con el trabajo previo y límites

El vector de 43 columnas conserva 25 rasgos históricos y los cinco rasgos de emojis, risas y polaridad con negación documentados previamente por Sebastián Sarmiento. Excluimos cuatro columnas históricas por su relación con la longitud o por redundancia. Los trece rasgos de este documento se añadieron después; `negated_positive_count` complementa `lex_pol_neg` sin reemplazarlo. La base del extractor procede del proyecto de Edwin Puertas y conserva su atribución.

Estas reglas no interpretan ironía ni entienden toda la conversación. Una sigla puede parecer una palabra enfática, una pregunta puede ser neutral y una negación puede describir un hecho sin expresar rechazo. Las listas léxicas tampoco cubren todas las variantes del español. Por eso estas características se usan junto a la representación del texto, no como una decisión de sentimiento por sí solas.
