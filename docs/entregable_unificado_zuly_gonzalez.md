# Entregable Unificado de Cambios - Zuly Gonzalez

## Alcance del documento

Este documento unifica en un solo archivo los cuatro frentes solicitados:

1. Caracteristicas implementadas.
2. Informe tecnico corto.
3. Bitacora de contribuciones.

El contenido esta limitado unicamente a cambios reales implementados por Zuly Gonzalez en este repositorio durante esta iteracion.

Queda fuera de alcance todo lo existente en la linea base previa del proyecto.

## Fuente de informacion y validacion

La informacion reportada en este documento proviene de evidencia directa del repositorio:

1. Codigo fuente actualizado en archivos de logica, pruebas y scripts.
2. Ejecucion de pruebas automatizadas del modulo en espanol.
3. Ejecucion del flujo completo con arranque_entregable.ps1.
4. Archivos CSV generados por procesamiento real del dataset TASS.

---

## Caracteristicas implementadas

Se implementaron y validaron tres caracteristicas lexicas nuevas orientadas a tweets en espanol:

1. Dudas (doubt_count).
2. Regionalismos (regionalism_count).
3. Intensificadores (intensifier_count).

### 1.1 Ubicacion tecnica

- Extraccion de features nuevas: logic/feature_extraction.py
- Lexico asociado: logic/lexical_features.py
- Pruebas del modulo: tests/test_nuevas_features.py

### 1.2 Comportamiento implementado

- Dudas:
  - Cuenta frases de duda en texto crudo.
  - Soporta variaciones con y sin tildes por normalizacion.

- Regionalismos:
  - Cuenta ocurrencias por token usando lexico en espanol.
  - Incluye normalizacion para coincidencia robusta.

- Intensificadores:
  - Cuenta intensificadores por token.
  - Aplica la misma normalizacion robusta de texto.

### 1.3 Integracion con vector

El vector de salida de nuevas caracteristicas contiene 8 posiciones:

1. emoji_count
2. emoji_pol
3. laugh_count
4. laugh_len
5. lex_pol_neg
6. doubt_count
7. regionalism_count
8. intensifier_count

### 1.4 Tabla de metodos creados y su funcion

| Metodo | Archivo | Funcion |
| --- | --- | --- |
| load_senticon(path) | logic/feature_extraction.py | Carga el lexico de polaridad (ML-SentiCon) para calcular polaridad lexical. |
| get_features_nuevas(message) | logic/feature_extraction.py | Orquesta el calculo del vector de 8 caracteristicas nuevas para cada tweet. |
| emoji_pol(message, lexical) | logic/feature_extraction.py | Cuenta emojis y calcula polaridad agregada segun listas positiva/negativa. |
| laugh_count(tokens_text, lexical) | logic/feature_extraction.py | Detecta risas y obtiene cantidad y longitud maxima. |
| lex_pol_neg(tokens_text, lexical, senticon) | logic/feature_extraction.py | Calcula polaridad media con alcance de negacion y corte por adversativas/puntuacion. |
| _normalize_text_for_match(text) | logic/feature_extraction.py | Normaliza tildes, mayusculas y espacios para emparejamiento consistente. |
| lexical_word_count(message, lexical_words) | logic/feature_extraction.py | Cuenta ocurrencias de terminos de lexico por token sobre texto normalizado. |
| doubt_count(message, lexical) | logic/feature_extraction.py | Cuenta frases de duda (hedges) usando busqueda robusta por patron. |
| process_split(path_in, path_out, fe) | scripts/procesar_tweets_features.py | Lee CSV, calcula features nuevas por tweet y guarda salida enriquecida. |
| main() | scripts/procesar_tweets_features.py | Ejecuta procesamiento train/test y reporta metricas resumen de las 3 nuevas caracteristicas. |

---

## Informe tecnico de implementacion

### 2.1 Archivos funcionales agregados o ajustados

- arranque_entregable.ps1
- scripts/procesar_tweets_features.py
- logic/feature_extraction.py
- logic/lexical_features.py
- tests/test_nuevas_features.py

### 2.2 Flujo de ejecucion automatizado

Se dejo un flujo reproducible de ejecucion:

1. Ejecutar pruebas.
2. Procesar TASS train/test.
3. Generar CSVs con features nuevas.

Salida generada por el proceso:

- data/tass/tass2018_es_train_features.csv
- data/tass/tass2018_es_test_features.csv

### 2.3 Validacion ejecutada

Validacion efectiva en esta iteracion:

- Pruebas de nuevas features en espanol: 16 passed.
- Arranque integral con script: ejecucion correcta sin errores.
- Procesamiento TASS completado:
  - tass2018_es_train_features.csv: 1008 tweets.
  - tass2018_es_test_features.csv: 506 tweets.

Promedios observados en train para nuevas caracteristicas:

- doubt_count: 0.0813
- regionalism_count: 0.0466
- intensifier_count: 0.1647

---

## Bitacora de contribuciones de Zuly Gonzalez

### 3.1 Implementacion

- Integracion de las tres caracteristicas en el extractor de features nuevas.
- Extension del lexico en espanol para dudas, regionalismos e intensificadores.
- Conservacion de normalizacion robusta para coincidencia textual.

### 3.2 Pruebas

- Ajuste de pruebas para validar vector de 8 caracteristicas.
- Pruebas con tweets reales del corpus TASS para las tres nuevas medidas.

### 3.3 Operacion

- Script de arranque integral para pruebas y procesamiento.
- Script de procesamiento para crear CSVs enriquecidos con codificacion utf-8-sig.

### 3.4 Consolidacion de enfoque

- El resultado final se dejo orientado solo a espanol.
- Se removio soporte en ingles para cumplir el alcance definido en esta fase.

---

La responsabilidad de los cambios aplicados y su validacion final corresponde a Zuly Gonzalez.
