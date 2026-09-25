# Característica léxica — Extractor Interjección "Eh"

**Característica:** recuento de la interjección 'eh' y sus variantes expresivas (`eh_count`).  
Implementación: clase `EhCountFeature` en `logic/eh_count_feature.py`.  
Pruebas: `tests/test_eh_count_feature.py` (8 pruebas con tweets reales del train).

## Hipótesis lingüística

La interjección 'eh' (junto a sus variantes fonéticas/expresivas como *ehhh*, *eh?* o *¡eh!*) cumple una función eminentemente fática, apelativa y enfática en el discurso informal. Se utiliza para captar la atención del interlocutor, demandar confirmación o remarcar estados emocionales ligados a la **queja, el reclamo, la sorpresa o la reafirmación** ("A ver si aprendemos, eh", "Ojo con esto, ehhh"). 

En minería de opiniones en X (Twitter), la presencia de 'eh' es un marcador claro de **informalidad conversacional y subjetividad**, estando prácticamente ausente en tuits descriptivos o neutros (**NONE**). Además, suele sesgarse hacia la **polaridad negativa (N)** (reproches y desacuerdos) o afirmativa/enfática (**P**). Dado que los tokenizadores estándar y la Bolsa de Palabras (BoW) suelen eliminar signos, considerar 'eh' como stopword o separar 'eh' de 'ehhh', esta característica personalizada extrae explícitamente dicho valor discursivo. Es una característica **distinta** de las 7 del extractor base y de las de los demás integrantes.

## Cálculo (precisión + cobertura)

Cuenta la presencia de la interjección combinando expresiones regulares y reglas de exclusión:

1. **Patrón de coincidencia flexible:** Uso de expresiones regulares (`\b[eE]+[hH]+\b`) para capturar el token independiente y sus alargamientos expresivos (*eh*, *ehh*, *EHHH*), independientemente de mayúsculas/minúsculas.
2. **Límites de palabra y signos:** Contempla la presencia de signos adjuntos (`¿eh?`, `¡eh!`, `eh...`) sin perder la coincidencia.
3. **Filtro de falsos positivos y ruido:**
   - Excluye coincidencias internas dentro de otras palabras (e.g., *vehículo*, *rehab*, *fehaciente*, *Ehrenberg*).
   - Limpieza previa de URLs, menciones (`@usuario`) y hashtags (`#EHF`) para evitar capturar fragmentos de código o nombres de usuario.

## Verificación en el train (1008 tweets)

| | N | NEU | NONE | P | Tweets con valor ≠ 0 |
|---|---|---|---|---|---|
| `eh_count` (media) | **0.048** | 0.012 | **0.004** | 0.031 | 3.8 % |

- **Diferenciación de NONE (0.004):** Es casi inexistente en tuits sin carga objetiva/neutra; su aparición es una señal clara de que el tuit contiene una postura o emoción activa.
- **Sesgo hacia la polaridad Negativa (0.048):** Predomina en tuits etiquetados como **N**, reflejando su uso frecuente en quejas, sarcasmo o interpelaciones en discusiones de Twitter.

## Ejemplos reales (train)

- `eh_count = 1` → *"A ver si nos enteramos de una vez, **eh**… que no todo vale."* (N)
- `eh_count = 1` → *"Ojo con el partidazo que se viene hoy, **ehhh**! VAMOOOS"* (P)

## Limitaciones

- **Baja cobertura (3.8 %):** Al ser un rasgo conversacional específico, aparece en un porcentaje reducido del corpus, actuando como característica de soporte y no como clasificador aislado.
- **Ruido en deformaciones tipográficas extremas:** Variaciones muy atípicas u omisiones de espacio (ej. *"pero_eh_mira"*) pueden requerir ajustes en el tokenizador previo.
- **No distingue tono por sí sola:** No diferencia explícitamente entre la apelación afectiva/entusiasta y la queja despectiva sin el contexto de las palabras circundantes.