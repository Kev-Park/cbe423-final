import pickle
import numpy as np
from pymatgen.core import Structure
import pandas as pd
from tqdm import tqdm


def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    structures = []
    targets = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Parsing {input_path}"):
        structure = Structure.from_str(row['cif'], fmt='cif')
        structures.append(structure)
        targets.append(row['formation_energy_per_atom'])

    targets = np.asarray(targets, dtype=np.float64)

    data = {
        'structures': structures,
        'targets': targets.astype(np.float32).tolist(),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    train_csv = 'raw_data/train.csv'
    val_csv = 'raw_data/val.csv'
    test_csv = 'raw_data/test.csv'

    preprocess_data(train_csv, 'preprocessed_data/train.pkl')
    preprocess_data(val_csv, 'preprocessed_data/val.pkl')
    preprocess_data(test_csv, 'preprocessed_data/test.pkl')
