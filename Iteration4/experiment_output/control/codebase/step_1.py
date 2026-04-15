# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np

def main():
    input_file = "/home/node/work/projects/materials_project_v3/perovskite_data.csv"
    output_file = "data/cleaned_perovskite_data.csv"
    df = pd.read_csv(input_file)
    cols_to_remove = ['surface_energy', 'work_function', 'A_period', 'B_period', 'radius_ratio']
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])
    df = df.loc[:, ~df.columns.duplicated()]
    cols_to_drop = [c for c in df.columns if c.endswith('.1') or c.endswith('.2')]
    df = df.drop(columns=cols_to_drop)
    VEC_DICT = {'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2, 'Sc': 3, 'Y': 3, 'La': 3, 'Ce': 4, 'Pr': 5, 'Nd': 6, 'Pm': 7, 'Sm': 8, 'Eu': 9, 'Gd': 10, 'Tb': 11, 'Dy': 12, 'Ho': 13, 'Er': 14, 'Tm': 15, 'Yb': 16, 'Lu': 17, 'Ti': 4, 'Zr': 4, 'Hf': 4, 'V': 5, 'Nb': 5, 'Ta': 5, 'Cr': 6, 'Mo': 6, 'W': 6, 'Mn': 7, 'Tc': 7, 'Re': 7, 'Fe': 8, 'Ru': 8, 'Os': 8, 'Co': 9, 'Rh': 9, 'Ir': 9, 'Ni': 10, 'Pd': 10, 'Pt': 10, 'Cu': 11, 'Ag': 11, 'Au': 11, 'Zn': 12, 'Cd': 12, 'Hg': 12, 'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3, 'C': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4, 'N': 5, 'P': 5, 'As': 5, 'Sb': 5, 'Bi': 5, 'O': 6, 'S': 6, 'Se': 6, 'Te': 6, 'Po': 6, 'F': 7, 'Cl': 7, 'Br': 7, 'I': 7, 'At': 7, 'He': 8, 'Ne': 8, 'Ar': 8, 'Kr': 8, 'Xe': 8, 'Rn': 8}
    def calculate_vec(row):
        a_site = row['A_site']
        b_site = row['B_site']
        return VEC_DICT.get(a_site, 0) + VEC_DICT.get(b_site, 0) + 3 * 6
    df['VEC'] = df.apply(calculate_vec, axis=1)
    r_O = df['B_radius'] / df['mu']
    V_geom = 8 * (df['A_radius'] + r_O)**2 * (df['B_radius'] + r_O)
    df['volume_residual'] = df['volume'] - V_geom
    Z = df['nsites'] / 5
    V_ideal_cell = 8 * (df['B_radius'] + r_O)**3 * Z
    df['tilt_proxy'] = df['volume'] / V_ideal_cell
    df['volume'] = np.log1p(df['volume'])
    mask = df['elastic_anisotropy'].notnull()
    df.loc[mask, 'elastic_anisotropy'] = np.log1p(np.clip(df.loc[mask, 'elastic_anisotropy'], 0, None))
    print("Data Cleaning and Feature Engineering Summary\n" + "="*45)
    print("Cleaned dataset shape: " + str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns\n")
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    print("Null counts per column (only showing columns with NaNs):")
    if len(null_cols) > 0:
        print(null_cols.to_string() + "\n")
    else:
        print("None\n")
    engineered_features = ['VEC', 'volume_residual', 'tilt_proxy', 'volume', 'elastic_anisotropy']
    print("Descriptive statistics for engineered features:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df[engineered_features].describe().to_string() + "\n")
    df.to_csv(output_file, index=False)
    print("Cleaned dataset saved to " + output_file)

if __name__ == '__main__':
    main()