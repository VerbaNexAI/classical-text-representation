# Documentación — Fase 1: nuevas características léxicas

Este documento describe las 4 características léxicas nuevas agregadas en `logic/feature_extraction.py` sobre el baseline del taller. A diferencia de una tabla por grupo temático, aquí cada característica se documenta como una sección independiente: qué problema lingüístico ataca, cómo está implementada, y qué evidencia respalda que no es constante ni redundante con lo que ya medía `get_features_lexical()`.

---

## `lex_pol_neg`

**Qué mide.** Un puntaje neto de polaridad léxica que resuelve el alcance de la negación: si una palabra del léxico de polaridad (`polaridad_es`) cae después de un marcador como "no" o "nunca" y antes de un cierre de alcance ("pero", "y", fin del tweet), su signo se invierte.

**Por qué hace falta.** El taller ya cuenta ocurrencias de `adjetives_pos`/`adjetives_neg`, pero un conteo crudo trata "no me gusta" igual que "me gusta": ambos suman un punto a positivo. `lex_pol_neg` es la única característica del módulo que resuelve ese caso.

**Evidencia de que no es trivial.** Sobre el corpus completo de train, se activa (da un valor distinto de 0) en el **26.5%** de los tweets — la cobertura más alta de las 4 características nuevas. En la ablación de la Fase 4 es también la que más aporta al macro-F1 del modelo final (+0.0031), consistente con esa mayor cobertura.

**Límite conocido, documentado con evidencia real:** sobre los 10 tweets del análisis de errores de la Fase 4, `lex_pol_neg` da 0.0 en los 10 casos, porque ninguna de las palabras relevantes ("encantado", "enamora", "estupendo", "adorable"...) está en el léxico de ~25 palabras por polaridad — es una limitación de cobertura del diccionario, no un bug de la lógica de negación (que sí se prueba por separado con casos donde la palabra sí está en el léxico).

---

## `elongated_words_count`

**Qué mide.** Cuántas palabras del tweet tienen una letra repetida 3 o más veces seguidas ("buenooo", "nooo").

**Por qué hace falta.** Es un marcador ortográfico de énfasis emocional propio del registro informal de Twitter que no tiene ningún equivalente en las 29 características originales del taller.

**Evidencia de que no es trivial.** Se activa en el **3.0%** del corpus — es la característica menos frecuente de las 4, lo cual es esperable (la elongación es un recurso puntual, no constante).

**Límite conocido:** no detecta risas tipo "jajajaja", porque ese patrón alterna caracteres distintos (j, a, j, a...) en vez de repetir uno solo — se verifica explícitamente en las pruebas unitarias que este es el comportamiento esperado, no un bug.

---

## `punct_repetition`

**Qué mide.** Rachas de 2 o más signos de exclamación (`!`/`¡`) o interrogación (`?`/`¿`) seguidos.

**Por qué hace falta.** BoW/TF-IDF tokenizan cada signo de puntuación por separado; no hay forma de que esas representaciones capturen que "!!!" es una señal de intensidad distinta a un solo "!".

**Evidencia de que no es trivial.** Se activa en el **4.9%** del corpus.

**Dependencia de implementación:** a diferencia de `lex_pol_neg`, que trabaja sobre el texto ya tokenizado, `elongated_words_count` y `punct_repetition` reciben el texto **crudo** (antes de `TextProcessing.transformer()`), porque `remove_patterns()` elimina toda la puntuación y normaliza el texto antes de tokenizar — si se les pasara el texto limpio, siempre darían 0.

---

## `lexical_diversity_mattr`

**Qué mide.** Diversidad de vocabulario (tipo-token ratio) calculada con ventana móvil de tamaño fijo, en vez de sobre el tweet completo.

**Por qué reemplaza a `lexical_diversity` (no la complementa).** La auditoría de Fase 0 encontró dos problemas en la característica original: (1) correlacionaba con la longitud del texto (r≈-0.81), es decir no era una señal de diversidad léxica limpia sino en buena parte un proxy de cuántas palabras tiene el tweet; y (2) por un bug de implementación medía diversidad de **caracteres**, no de **palabras**. MATTR corrige ambos problemas: al promediar el TTR sobre ventanas fijas, dos tweets de distinta longitud son comparables entre sí, y se calcula sobre tokens de palabra, no caracteres.

**Por qué no tiene una "tasa de activación":** a diferencia de las otras 3, no es una característica de presencia/ausencia sino un valor continuo entre 0 y 1 para todo tweet no vacío — la evidencia relevante aquí no es cobertura sino que ya no reproduce la correlación con la longitud que tenía la versión original (verificado en la auditoría de Fase 0, no en esta característica de reemplazo directamente, ya que MATTR es matemáticamente insensible a la longitud por construcción de la ventana móvil).

---

## Resumen

| Característica | Tipo | Requiere texto crudo | Cobertura / comportamiento | Aporte individual en ablación |
|---|---|---|---|---|
| `lex_pol_neg` | Presencia/signo | No (texto tokenizado) | 26.5% de los tweets | +0.0031 (la mayor de las 4) |
| `elongated_words_count` | Conteo | Sí | 3.0% de los tweets | < 0.004 (dentro del ruido) |
| `punct_repetition` | Conteo | Sí | 4.9% de los tweets | < 0.004 (dentro del ruido) |
| `lexical_diversity_mattr` | Continua [0,1] | No (tokens) | Reemplaza a `lexical_diversity`, sin su sesgo de longitud | No aislada individualmente en la ablación (ver Fase 4) |

## Pruebas

Las 4 características tienen pruebas unitarias en `tests/test_lexical_features.py`, con casos construidos para aislar el fenómeno lingüístico de cada una y, para `lex_pol_neg`, una prueba parametrizada sobre los 10 tweets reales del análisis de errores de la Fase 4 que documenta el límite de cobertura del léxico descrito arriba.
