# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    input_path = '/home/node/work/projects/materials_project_v3/perovskite_data.csv'
    output_dir = 'data/'
    output_path = os.path.join(output_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(input_path)
    print('========================================')
    print('--- BEFORE CLEANING ---')
    print('========================================')
    print('Shape: ' + str(df.shape))
    print('\nMissing values per column:')
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0].sort_values(ascending=False))
    print('\nDistributions (numerical columns):')
    print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])
    cols_to_drop = ['surface_energy', 'work_function', 'A_period', 'B_period', 'theoretical']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    duplicate_cols = [col for col in df.columns if col.endswith('.1')]
    df = df.drop(columns=duplicate_cols)
    if 'material_id' in df.columns:
        df = df.set_index('material_id')
    if 'volume' in df.columns:
        df['log_volume'] = np.log1p(df['volume'])
        df = df.drop(columns=['volume'])
    if 'elastic_anisotropy' in df.columns:
        df['log_elastic_anisotropy'] = np.log1p(df['elastic_anisotropy'].clip(lower=0))
        df = df.drop(columns=['elastic_anisotropy'])
    if 'tau' in df.columns:
        df['abs_tau_diff'] = np.abs(df['tau'] - 1.0)
    if 'A_radius' in df.columns and 'B_radius' in df.columns:
        df['radius_diff'] = df['A_radius'] - df['B_radius']
    if 'A_ie1' in df.columns and 'B_ie1' in df.columns:
        df['ie_ratio'] = df['A_ie1'] / df['B_ie1']
    if 'radius_ratio' in df.columns:
        df = df.drop(columns=['radius_ratio'])
    if 'K_VRH' in df.columns:
        df['is_elastic_characterized'] = df['K_VRH'].notnull()
    print('\n========================================')
    print('--- AFTER CLEANING ---')
    print('========================================')
    print('Shape: ' + str(df.shape))
    print('\nMissing values per column:')
    missing_after = df.isnull().sum()
    print(missing_after[missing_after > 0].sort_values(ascending=False))
    print('\nDistributions (numerical columns):')
    print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])
    full_count = len(df)
    elastic_count = df['is_elastic_characterized'].sum()
    print('\n========================================')
    print('--- DATASET COUNTS ---')
    print('========================================')
    print('Total rows in full dataset: ' + str(full_count))
    print('Rows in elastic subset (characterized): ' + str(elastic_count))
    df.to_csv(output_path)
    print('\nCleaned dataset saved to ' + output_path)