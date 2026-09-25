# classical-text-representation
Enfoques clásicos de representación de texto para PLN: Bolsa de Palabras, TF-IDF, n-gramas, Fastext, Wor2Vec

## Procesar tweets con 3 nuevas caracteristicas

## Arranque completo para entregables

Script recomendado (ejecuta pruebas y procesamiento de tweets):

```powershell
.\arranque_entregable.ps1
```

Opciones utiles:

```powershell
# Solo procesamiento
.\arranque_entregable.ps1 -SkipTests

# Solo pruebas
.\arranque_entregable.ps1 -SkipProcessing
```

Se implementaron estas caracteristicas lexicas:

1. Dudas (`doubt_count`): expresiones como "no se", "puede que", "no estoy seguro".
2. Regionalismos (`regionalism_count`): expresiones como "sumercé/sumerce", "parce", "bacano", "vale".
3. Intensificadores (`intensifier_count`): expresiones como "super/súper", "muy".

La evidencia de autoria, informe, bitacora y declaracion de uso de IA orientada solo a cambios de Zuly Gonzalez se consolida en:
- `docs/entregable_unificado_zuly_gonzalez.md`

Ejecucion desde consola (PowerShell):

```powershell
.\.venv\Scripts\python scripts\procesar_tweets_features.py
```

Salida:

- `data/tass/tass2018_es_train_features.csv`
- `data/tass/tass2018_es_test_features.csv`

## Entregables

- Notebook ejecutado end-to-end (BoW, TF-IDF, lexico, union, seleccion de clasificador, evaluacion, ablacion, errores):
	- `notebooks/entregable_bow_tfidf_lexico_union.ipynb`
- Documento unificado (caracteristicas, informe, bitacora y uso de IA):
	- `docs/entregable_unificado_zuly_gonzalez.md`
