# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

def clean_and_engineer_features():
    input_path = "/home/node/work/projects/materials_project_v3/perovskite_data.csv"
    data_dir = "data"
    output_path = os.path.join(data_dir, "cleaned_perovskite_data.csv")
    df = pd.read_csv(input_path)
    print("Original dataset shape: " + str(df.shape))
    cols_to_drop = ['surface_energy', 'work_function', 'A_period', 'B_period']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    print("Dropped zero-variance columns: " + ", ".join(existing_cols_to_drop))
    duplicate_cols = [c for c in df.columns if c.endswith('.1')]
    if duplicate_cols:
        df = df.drop(columns=duplicate_cols)
        print("Dropped duplicate columns: " + ", ".join(duplicate_cols))
    if 'volume' in df.columns:
        df['log_volume'] = np.log1p(df['volume'])
        print("Applied log1p transformation to 'volume'.")
    if 'elastic_anisotropy' in df.columns:
        df['log_elastic_anisotropy'] = np.log1p(df['elastic_anisotropy'].clip(lower=0))
        print("Applied log1p transformation to 'elastic_anisotropy'.")
    glazer_mapping = {221: 'a0a0a0', 62: 'a+b-b-', 167: 'a-a-a-', 140: 'a0a0c-', 74: 'a0b-b-', 127: 'a0a0c+', 15: 'a-b-b-', 14: 'a-b-c-', 137: 'a0a0c-', 161: 'a-a-a-'}
    if 'spacegroup_number' in df.columns:
        df['glazer_tilt'] = df['spacegroup_number'].map(glazer_mapping).fillna('Other')
        print("Mapped 'spacegroup_number' to 'glazer_tilt'.")
        tilt_counts = df['glazer_tilt'].value_counts().to_dict()
        print("Glazer tilt system distribution: " + str(tilt_counts))
    if 'tau' in df.columns:
        df['tau_strain'] = np.abs(df['tau'] - 1.0)
        print("Computed 'tau_strain' proxy.")
    if 'mu' in df.columns:
        df['mu_strain'] = np.abs(df['mu'] - 0.57)
        print("Computed 'mu_strain' proxy.")
    df.to_csv(output_path, index=False)
    print("Cleaned dataset saved to " + output_path)
    print("Final dataset shape: " + str(df.shape))

if __name__ == '__main__':
    clean_and_engineer_features()