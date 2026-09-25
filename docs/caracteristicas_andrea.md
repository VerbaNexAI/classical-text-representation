# Característica léxica — Andrea Jinet Nova Osorio

**Característica:** diminutivos afectivos (`diminutive_count`).
Implementación: clase `DiminutiveFeature` en `logic/diminutive_feature.py`.
Pruebas: `tests/test_diminutive_feature.py` (8 pruebas con tweets reales del train).

## Hipótesis lingüística

El diminutivo en español no siempre reduce el tamaño: con frecuencia expresa **afecto, cercanía o cortesía**
('un ratito', 'pobrecito', 'despacito'). Es un marcador de subjetividad afectiva que **escasea en los tuits
sin carga (NONE)** y tiende a acompañar polaridad **positiva**. La Bolsa de Palabras ve 'casa' y 'casita'
como tokens distintos y no captura ese matiz. Es una característica **distinta** de las 7 del extractor base
y de las de los demás integrantes.

## Cálculo (precisión + cobertura)

Cuenta los diminutivos afectivos combinando tres mecanismos para que "no se escape" nada:

1. **Lista curada** de diminutivos afectivos frecuentes en tuits (`besito`, `ratito`, `pobrecito`…).
2. **Regla morfológica** para los sufijos productivos `-ito/-ita/-itos/-itas` y `-illo/-illa/-illos/-illas`.
3. **Lista de exclusión** de falsos positivos: verbos (`necesito`, `repito`), adjetivos/sustantivos que
   terminan igual pero no son diminutivos (`bonito`, `infinito`, `pastilla`, `semilla`, `maravilla`) y
   topónimos (`Sevilla`). Además se eliminan menciones (@), hashtags (#) y URLs, y se comparan las palabras
   en minúscula y sin acentos.

## Verificación en el train (1008 tweets)

| | N | NEU | NONE | P | Tweets con valor ≠ 0 |
|---|---|---|---|---|---|
| `diminutive_count` (media) | 0.055 | 0.053 | **0.022** | 0.054 | 4.9 % |

- Distingue **NONE** (0.022, la más baja) del resto: el afecto aparece en tuits con carga, no en los neutros.
- Las clases con opinión (N, NEU, P) tienen medias parecidas y más altas; combinada con otros rasgos ayuda
  a separar "hay postura" de "no hay postura".

## Ejemplos reales (train)

- `diminutive_count = 1` → *"Me dice un **pajarito** que ahora mismo no puede reservar…"*.
- `diminutive_count = 1` → *"…acabo de hablar con alba y ya estoy un **poquito** mejor…"* (P).

## Limitaciones

- Cobertura baja (4.9 %): es una señal de apoyo, no un clasificador por sí sola.
- La regla morfológica puede dejar pasar algún nombre propio en diminutivo (`Karlita`) o requerir ampliar la
  lista de exclusión ante formas nuevas.
- No distingue el diminutivo afectivo del despectivo (`abogadillo`), poco frecuente en el corpus.
