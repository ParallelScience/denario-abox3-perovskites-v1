# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

def main():
    file_path = '/home/node/work/projects/materials_project_v3/perovskite_data.csv'
    df = pd.read_csv(file_path)
    cols_to_drop = ['surface_energy', 'work_function', 'A_period', 'B_period']
    duplicate_cols = [col for col in df.columns if col.endswith('.1')]
    df.drop(columns=cols_to_drop + duplicate_cols, inplace=True, errors='ignore')
    df['is_stable_soft'] = df['energy_above_hull'] < 0.05
    vec_map = {'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2, 'Sc': 3, 'Y': 3, 'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3, 'Ti': 4, 'Zr': 4, 'Hf': 4, 'V': 5, 'Nb': 5, 'Ta': 5, 'Cr': 6, 'Mo': 6, 'W': 6, 'Mn': 7, 'Tc': 7, 'Re': 7, 'Fe': 8, 'Ru': 8, 'Os': 8, 'Co': 9, 'Rh': 9, 'Ir': 9, 'Ni': 10, 'Pd': 10, 'Pt': 10, 'Cu': 11, 'Zn': 12, 'Al': 3, 'Ga': 3, 'In': 3, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4, 'P': 5, 'As': 5, 'Sb': 5, 'Bi': 5}
    df['VEC'] = df['A_site'].map(vec_map) + df['B_site'].map(vec_map) + 18
    r_O = df['B_radius'] / df['mu']
    V_geom = (df['A_radius'] + r_O)**2 * (df['B_radius'] + r_O) * 8
    df['volume_residual'] = df['volume'] - V_geom
    df['volume'] = np.log1p(df['volume'])
    df['elastic_anisotropy'] = np.log1p(df['elastic_anisotropy'].clip(lower=0))
    if 'radius_ratio' in df.columns:
        df.drop(columns=['radius_ratio'], inplace=True)
    output_path = os.path.join('data', 'cleaned_perovskite_data.csv')
    df.to_csv(output_path, index=False)
    print('Data Preprocessing and Feature Engineering Completed.')
    print('Final dataset shape: ' + str(df.shape))
    print('\nTarget \'is_stable_soft\' distribution:\n' + str(df['is_stable_soft'].value_counts(normalize=True)))
    print('\nVEC summary statistics:\n' + str(df['VEC'].describe()))
    print('\nVolume Residual summary statistics:\n' + str(df['volume_residual'].describe()))
    print('\nLog-transformed Volume summary statistics:\n' + str(df['volume'].describe()))
    print('\nLog-transformed Elastic Anisotropy summary statistics:\n' + str(df['elastic_anisotropy'].describe()))
    print('\nCleaned dataset saved to: ' + output_path)

if __name__ == '__main__':
    main()