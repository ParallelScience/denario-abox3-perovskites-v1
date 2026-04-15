# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    csv_path = '/home/node/work/projects/materials_project_v3/perovskite_data.csv'
    df = pd.read_csv(csv_path)
    if 'material_id' in df.columns:
        df.set_index('material_id', inplace=True)
    cols_to_drop = [col for col in df.columns if col.endswith('.1')]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]
    cols_to_remove = ['surface_energy', 'work_function', 'A_period', 'B_period']
    df.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    if 'volume' in df.columns:
        df['volume'] = np.log1p(df['volume'])
    if 'elastic_anisotropy' in df.columns:
        df['elastic_anisotropy'] = np.log1p(df['elastic_anisotropy'])
    if 'tau' in df.columns:
        df['abs_tau_diff'] = np.abs(df['tau'] - 1.0)
    if 'A_radius' in df.columns and 'B_radius' in df.columns:
        df['radius_diff'] = df['A_radius'] - df['B_radius']
    if 'A_ie1' in df.columns and 'B_ie1' in df.columns:
        df['ie_ratio'] = df['A_ie1'] / df['B_ie1']
    if 'radius_ratio' in df.columns:
        df.drop(columns=['radius_ratio'], inplace=True)
    print('Cleaned Dataset Shape: ' + str(df.shape))
    print('\nColumn List:')
    print(list(df.columns))
    print('\nNull Counts per Column:')
    pd.set_option('display.max_rows', None)
    print(df.isnull().sum().to_string())
    pd.reset_option('display.max_rows')
    print('\nDescriptive Statistics for Engineered Features:')
    engineered_cols = ['abs_tau_diff', 'radius_diff', 'ie_ratio']
    print(df[engineered_cols].describe().to_string())
    output_path = 'data/cleaned_perovskite_data.csv'
    df.to_csv(output_path)
    print('\nCleaned dataset saved to ' + output_path)