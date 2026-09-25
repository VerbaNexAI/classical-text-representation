# Características léxicas de Ivan Cruz y Juan Ochoa

Módulo: `logic/IvanCruz_JuanOchoa_features.py`
Tests:  `tests/test_IvanCruz_JuanOchoa_features.py`

Aporte de **Ivan Cruz** y **Juan Ochoa** al proyecto grupal de clasificación de
sentimiento sobre TASS 2018. Contiene 9 características léxicas nuevas,
organizadas en 3 grupos (expresividad, modificadores, coloquial), todas
vectorizadas dentro del módulo y probadas con tweets reales del corpus de
entrenamiento.

## Hipótesis lingüística por característica

### Grupo: expresividad

| Característica | Hipótesis | Método |
|---|---|---|
| `elongated_count` | Alargar letras ("buenooo", "noooo") es un sustituto escrito de la prosodia y marca énfasis emocional, asociado a polaridad fuerte. | Cuenta palabras con 3+ letras repetidas seguidas. Ignora números y siglas en mayúsculas de ≤3 letras ("CCC", "III"). |
| `upper_ratio` | Palabras completas en mayúsculas equivalen a gritar. | Proporción de palabras con ≥2 letras que están 100% en mayúsculas, sobre el total de palabras elegibles. Excluye "RT". |
| `exclam_run_max` | Repetir "!" amplifica la emoción; la longitud de la racha mide intensidad, no solo presencia. | Longitud de la racha más larga de `!` o `¡`. |
| `question_run_max` | Repetir "?" suele indicar incredulidad o molestia. | Longitud de la racha más larga de `?` o `¿`. |
| `interrobang_count` | La combinación "?!" / "!?" expresa sorpresa o indignación mezcladas, distinta de solo "!" o "?". | Cuenta ocurrencias de `!\s*\?` o `?\s*!`. |

### Grupo: modificadores

| Característica | Hipótesis | Método |
|---|---|---|
| `intensifier_count` | Intensificadores ("muy", "súper", "demasiado") refuerzan la polaridad del adjetivo o verbo que modifican. | Cuenta ocurrencias exactas de una lista de 21 intensificadores, con normalización de tildes y colapso de letras repetidas ("muuuy" → "muy"). |
| `attenuator_count` | Atenuadores ("algo", "un poco", "apenas") suavizan la valoración y son más comunes en opiniones matizadas o neutras. | Cuenta ocurrencias de una lista de 13 atenuadores (incluye frases de dos palabras: "un poco", "al menos"). |

### Grupo: coloquial

| Característica | Hipótesis | Método |
|---|---|---|
| `interjection_count` | Interjecciones ("ay", "uf", "wow", "ojalá") señalan reacción emocional inmediata. | Cuenta ocurrencias de una lista de 27 interjecciones. |
| `regionalism_count` | Marcadores dialectales ("wey", "chido", "mola", "guay") tienen su propio registro evaluativo. | Cuenta ocurrencias de una lista de 21 regionalismos. |

## Auditoría sobre el corpus de train (1008 tweets)

Se aplicó la misma función `audit_features()` de la Fase 0 a las 9 características:

| Feature | activation_rate | is_constant | corr con nº tokens | length_proxy |
|---|---|---|---|---|
| `elongated_count` | 0.061 | No | 0.18 | No |
| `upper_ratio` | 0.214 | No | 0.04 | No |
| `exclam_run_max` | 0.187 | No | 0.10 | No |
| `question_run_max` | 0.052 | No | 0.11 | No |
| `interrobang_count` | 0.034 | No | 0.13 | No |
| `intensifier_count` | 0.418 | No | 0.32 | No |
| `attenuator_count` | 0.291 | No | 0.28 | No |
| `interjection_count` | 0.152 | No | 0.19 | No |
| `regionalism_count` | 0.043 | No | 0.14 | No |

**Conclusiones de la auditoría:**

- Ninguna es constante en el corpus.
- Ninguna supera el umbral de proxy de longitud (|r| ≥ 0.7). El máximo es
  `intensifier_count` (r = 0.32), muy por debajo.
- Sensibilidad al preprocesamiento: `upper_ratio` pierde ~7 % de su señal si el
  texto se pasa a minúsculas antes de extraer. **Documentado explícitamente**: este
  módulo debe recibir texto crudo, no el que ya pasó por
  `TextProcessing.transformer()`.

## Diferencias con el diccionario del taller

- `intensifier_count` vs. `adverb_cant` del taller: el taller incluye "algo" en
  adverb_cant, aquí está en atenuadores. `adverb_cant` no incluye "súper",
  "totalmente", "demasiado", que sí están aquí.
- `attenuator_count` vs. `adverb_mode` del taller: el taller incluye "bien", "mal",
  "regular", que son valorativos, no atenuadores. Aquí no.
- `interjection_count` y `regionalism_count` no tienen equivalente en el taller.
- `elongated_count`, `upper_ratio`, `exclam_run_max`, `question_run_max`,
  `interrobang_count` tampoco tienen equivalente en el taller.

## Limitaciones

- Las listas de intensificadores, atenuadores, interjecciones y regionalismos se
  auditaron sobre este corpus (tweets de 2016, España). No hay garantía de que
  generalicen a otro dominio o época sin repetir la Fase 0.
- `regionalism_count` está sesgado hacia variedades con presencia en el corpus;
  en InterTASS-PE o InterTASS-CR habría que volver a auditar.
- `upper_ratio` y `elongated_count` requieren texto crudo; encadenarlas después
  del `TextProcessing.transformer()` las inutiliza parcialmente.

## Pruebas

`tests/test_IvanCruz_JuanOchoa_features.py` contiene pruebas parametrizadas con
tweets reales del corpus de entrenamiento, identificados por `tweetid`:

- Coincidencia con el CSV de train cuando está disponible.
- Estructura del módulo (9 features, 3 grupos).
- Robustez a entradas vacías / NaN / no-string.
- Casos reales para cada característica.
- Casos de regresión (siglas, "el Che", "vaya" verbo, "medio" no atenuador,
  minúsculas apagan `upper_ratio`).
- Auditoría: ninguna constante, ninguna proxy de longitud.