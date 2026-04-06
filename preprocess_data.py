import pickle
from pymatgen.core import Structure
import pandas as pd
from tqdm import tqdm

def preprocess_data(input_path, output_path):

    df = pd.read_csv(input_path)
    structures = []
    targets = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing CIF rows"):
        structure = Structure.from_str(row['cif'], fmt='cif')
        structures.append(structure)
        targets.append(row['e_above_hull'])
    
    data = {'structures': structures, 'targets': targets}
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
if __name__ == "__main__":
    input_path = 'raw_data/test.csv'
    output_path = 'preprocessed_data/test.pkl'
    preprocess_data(input_path, output_path)