# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

def main():
    input_filepath = '/home/node/work/projects/materials_project_v3/perovskite_data.csv'
    output_dir = 'data'
    output_filepath = os.path.join(output_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(input_filepath)
    print('Original dataset shape: ' + str(df.shape))
    zero_var_cols = ['surface_energy', 'work_function', 'A_period', 'B_period']
    df = df.drop(columns=zero_var_cols, errors='ignore')
    duplicate_cols = [col for col in df.columns if col.endswith('.1')]
    cols_to_drop = duplicate_cols + ['radius_ratio']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df['volume'] = np.log1p(df['volume'])
    if 'elastic_anisotropy' in df.columns:
        df['elastic_anisotropy'] = np.log1p(df['elastic_anisotropy'].clip(lower=0))
    vec_dict = {'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2, 'Sc': 3, 'Y': 3, 'La': 3, 'Ce': 4, 'Pr': 5, 'Nd': 6, 'Sm': 8, 'Eu': 9, 'Gd': 10, 'Tb': 11, 'Dy': 12, 'Ho': 13, 'Er': 14, 'Tm': 15, 'Yb': 16, 'Lu': 17, 'Ti': 4, 'Zr': 4, 'Hf': 4, 'V': 5, 'Nb': 5, 'Ta': 5, 'Cr': 6, 'Mo': 6, 'W': 6, 'Mn': 7, 'Tc': 7, 'Re': 7, 'Fe': 8, 'Ru': 8, 'Os': 8, 'Co': 9, 'Rh': 9, 'Ir': 9, 'Ni': 10, 'Pd': 10, 'Pt': 10, 'Cu': 11, 'Zn': 12, 'Al': 3, 'Ga': 3, 'In': 3, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4, 'P': 5, 'As': 5, 'Sb': 5, 'Bi': 5, 'O': 6}
    def get_vec(row):
        a_el = row.get('A_site')
        b_el = row.get('B_site')
        a_vec = vec_dict.get(a_el)
        if a_vec is None:
            group = row.get('A_group', 0)
            a_vec = group if group <= 12 else group - 10
        b_vec = vec_dict.get(b_el)
        if b_vec is None:
            group = row.get('B_group', 0)
            b_vec = group if group <= 12 else group - 10
        return a_vec + b_vec + 18
    df['VEC'] = df.apply(get_vec, axis=1)
    df = pd.get_dummies(df, columns=['magnetic_ordering', 'crystal_system'], drop_first=False)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    print('Cleaned dataset shape: ' + str(df.shape))
    df.to_csv(output_filepath, index=False)
    print('Cleaned dataset saved to ' + output_filepath)

if __name__ == '__main__':
    main()