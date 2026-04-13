import pickle
import numpy as np
from pymatgen.core import Structure
import pandas as pd
from tqdm import tqdm


def preprocess_data(input_path, output_path, target_mean, target_std):
    df = pd.read_csv(input_path)
    structures = []
    raw_targets = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Parsing {input_path}"):
        structure = Structure.from_str(row['cif'], fmt='cif')
        structures.append(structure)
        raw_targets.append(row['formation_energy_per_atom'])

    raw_targets = np.asarray(raw_targets, dtype=np.float64)
    targets = (raw_targets - target_mean) / target_std

    data = {
        'structures': structures,
        'targets': targets.astype(np.float32).tolist(),
        'raw_targets': raw_targets.astype(np.float32).tolist(),
        'target_mean': float(target_mean),
        'target_std': float(target_std),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def compute_train_stats(train_csv_path):
    df = pd.read_csv(train_csv_path)
    raw = df['formation_energy_per_atom'].to_numpy(dtype=np.float64)
    mean = raw.mean()
    std = raw.std()
    if std == 0:
        raise ValueError("Training target std is zero; cannot z-score.")
    return mean, std


if __name__ == "__main__":
    train_csv = 'raw_data/train.csv'
    val_csv = 'raw_data/val.csv'
    test_csv = 'raw_data/test.csv'

    mean, std = compute_train_stats(train_csv)
    print(f"Train target mean: {mean:.6f}, std: {std:.6f}", flush=True)

    with open('preprocessed_data/target_stats.pkl', 'wb') as f:
        pickle.dump({'target_mean': float(mean), 'target_std': float(std)}, f)

    preprocess_data(train_csv, 'preprocessed_data/train.pkl', mean, std)
    preprocess_data(val_csv, 'preprocessed_data/val.pkl', mean, std)
    preprocess_data(test_csv, 'preprocessed_data/test.pkl', mean, std)
