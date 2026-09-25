from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logic.feature_extraction import FeatureExtraction, NEW_FEATURE_NAMES


def process_split(path_in: Path, path_out: Path, fe: FeatureExtraction) -> pd.DataFrame:
    df = pd.read_csv(path_in)
    if 'content' not in df.columns:
        raise KeyError(f"El archivo {path_in} no tiene la columna 'content'.")

    feats = df['content'].fillna('').astype(str).map(fe.get_features_nuevas)
    feat_df = pd.DataFrame(feats.tolist(), columns=NEW_FEATURE_NAMES)
    out = pd.concat([df, feat_df], axis=1)
    out.to_csv(path_out, index=False, encoding='utf-8-sig')
    return out


def main():
    tass_dir = ROOT / 'data' / 'tass'
    train_in = tass_dir / 'tass2018_es_train.csv'
    test_in = tass_dir / 'tass2018_es_test.csv'

    if not train_in.exists() or not test_in.exists():
        raise FileNotFoundError(f"No se encontraron CSV en {tass_dir}")

    fe = FeatureExtraction('es')

    train_out = tass_dir / 'tass2018_es_train_features.csv'
    test_out = tass_dir / 'tass2018_es_test_features.csv'

    train_df = process_split(train_in, train_out, fe)
    test_df = process_split(test_in, test_out, fe)

    print('Procesamiento completado:')
    print(f' - {train_out.name}: {len(train_df)} tweets')
    print(f' - {test_out.name}: {len(test_df)} tweets')
    print('\nPromedios de las 3 nuevas caracteristicas:')
    cols = ['doubt_count', 'regionalism_count', 'intensifier_count']
    print(train_df[cols].mean().round(4).to_string())


if __name__ == '__main__':
    main()
