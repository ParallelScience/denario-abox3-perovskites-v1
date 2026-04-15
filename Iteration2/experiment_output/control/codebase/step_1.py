# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

def main():
    input_path = '/home/node/work/projects/materials_project_v3/perovskite_data.csv'
    output_dir = 'data'
    output_path = os.path.join(output_dir, 'cleaned_dataset.csv')
    print('Loading dataset from: ' + input_path)
    df = pd.read_csv(input_path)
    duplicate_cols = [col for col in df.columns if col.endswith('.1')]
    if duplicate_cols:
        print('Dropping duplicate columns: ' + ', '.join(duplicate_cols))
        df = df.drop(columns=duplicate_cols)
    cols_to_drop = ['surface_energy', 'work_function', 'A_period', 'B_period', 'theoretical']
    cols_to_drop_present = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop_present:
        print('Dropping irrelevant columns: ' + ', '.join(cols_to_drop_present))
        df = df.drop(columns=cols_to_drop_present)
    if 'material_id' in df.columns:
        df = df.set_index('material_id')
        print('Set material_id as index.')
    if 'volume' in df.columns:
        df['volume'] = np.log1p(df['volume'])
        print('Applied log1p transformation to volume.')
    if 'elastic_anisotropy' in df.columns:
        df['elastic_anisotropy'] = np.log1p(df['elastic_anisotropy'])
        print('Applied log1p transformation to elastic_anisotropy.')
    print('Engineering new features...')
    if 'tau' in df.columns:
        df['abs_tau_diff'] = np.abs(df['tau'] - 1.0)
    if 'A_radius' in df.columns and 'B_radius' in df.columns:
        df['radius_diff'] = df['A_radius'] - df['B_radius']
    if 'A_en' in df.columns and 'B_en' in df.columns:
        df['en_diff'] = np.abs(df['B_en'] - df['A_en'])
        df['en_var'] = df[['A_en', 'B_en']].var(axis=1)
    if 'A_valence' in df.columns:
        a_val = df['A_valence']
    else:
        a_val = df['A_group']
        print('Column A_valence not found. Using A_group for VEC calculation.')
    if 'B_valence' in df.columns:
        b_val = df['B_valence']
    else:
        b_val = df['B_group']
        print('Column B_valence not found. Using B_group for VEC calculation.')
    df['VEC'] = a_val + b_val + 6
    if 'radius_ratio' in df.columns:
        df = df.drop(columns=['radius_ratio'])
        print('Dropped radius_ratio to reduce multicollinearity.')
    cat_cols = ['crystal_system', 'magnetic_ordering', 'A_site', 'B_site']
    cat_cols_present = [col for col in cat_cols if col in df.columns]
    if cat_cols_present:
        print('One-hot encoding categorical columns: ' + ', '.join(cat_cols_present))
        df = pd.get_dummies(df, columns=cat_cols_present, drop_first=False, dtype=int)
    print('\n--- Final Dataset Summary ---')
    print('Final feature matrix shape: ' + str(df.shape))
    print('\nNull counts per column (showing only columns with nulls):')
    pd.set_option('display.max_rows', None)
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(null_cols.to_string())
    else:
        print('No missing values found.')
    df.to_csv(output_path)
    print('\nCleaned dataset saved to ' + output_path)

if __name__ == '__main__':
    main()