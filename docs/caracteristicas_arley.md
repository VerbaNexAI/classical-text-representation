# Característica léxica — Arley Rativa

**Característica:** léxico malsonante (`vulgarity_count`).
Implementación: clase `VulgarityFeature` en `logic/vulgarity_feature.py`.
Pruebas: `tests/test_vulgarity_feature.py` (7 pruebas con tweets reales del train).

## Hipótesis lingüística

El léxico malsonante (tacos, insultos, palabras soeces) es un marcador pragmático de **alta intensidad
emocional**, mayoritariamente **negativa**. La Bolsa de Palabras trata cada taco como un token aislado y
no captura que su presencia, en conjunto, es una señal de polaridad; agruparlos en un solo rasgo la hace
explícita. Es una característica **distinta** de las 7 del extractor base y de las de los demás integrantes
(emojis, risas, negación, interjección "eh").

## Cálculo

Cuenta cuántas palabras del tuit pertenecen a un léxico malsonante curado. Para que "no se escape" nada:

- se comparan las palabras en **minúscula y sin acentos**;
- se **eliminan menciones (@), hashtags (#) y URLs** antes de contar (para no confundir nombres de usuario);
- se **colapsan las elongaciones** de 3+ caracteres iguales: `putooo → puto`, `joderrr → joder`;
- el léxico incluye variantes de género/número y formas frecuentes en varias variedades del español.

## Verificación en el train (1008 tweets)

| | N | NEU | NONE | P | Tweets con valor ≠ 0 |
|---|---|---|---|---|---|
| `vulgarity_count` (media) | **0.084** | 0.008 | 0.007 | 0.006 | 3.5 % |

- Es un discriminador **muy limpio de N**: la media en N (0.084) es ~10× la de cualquier otra clase.
- Cobertura baja (3.5 %), como toda señal léxica específica: aporta como complemento del BoW/TF-IDF.

## Ejemplos reales (train)

- `vulgarity_count = 1` → *"…que **puto** mal escribo…"* (N).
- `vulgarity_count = 2` → *"…me suelta la **puta subnormal**, te estoy llamando…"* (N).

## Limitaciones

- El taco puede aparecer en tono **cómplice o positivo** (*"es la ilusión de mi **puta** vida"*, P), así que
  no siempre implica polaridad negativa.
- Depende del léxico: términos malsonantes no listados (o muy censurados, `m*erda`) no se detectan.
