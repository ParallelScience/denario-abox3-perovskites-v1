# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    file_path = '/home/node/work/projects/materials_project_v3/perovskite_data.csv'
    print('Loading dataset from: ' + file_path + '\n')
    df = pd.read_csv(file_path)
    print('--- Dataset Shape ---')
    print('Rows: ' + str(df.shape[0]) + ', Columns: ' + str(df.shape[1]) + '\n')
    print('--- Column Names and Data Types ---')
    print(df.dtypes.to_string() + '\n')
    print('--- Missing Values (NaNs) ---')
    missing_values = df.isna().sum()
    missing_cols = missing_values[missing_values > 0]
    if len(missing_cols) > 0:
        print(missing_cols.to_string() + '\n')
    else:
        print('No missing values found.\n')
    print('--- Infinite Values ---')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        print(inf_cols.to_string() + '\n')
    else:
        print('No infinite values found.\n')
    print('--- Duplicate material_id entries ---')
    duplicates = df.duplicated(subset=['material_id']).sum()
    print('Number of duplicate material_id entries: ' + str(duplicates) + '\n')
    print('--- First 5 Rows ---')
    print(df.head(5).to_string() + '\n')
    print('--- Last 5 Rows ---')
    print(df.tail(5).to_string() + '\n')
    print('--- Unique Value Counts ---')
    cols_to_count = ['crystal_system', 'magnetic_ordering', 'A_site', 'B_site', 'spacegroup_symbol']
    for col in cols_to_count:
        print('Counts for ' + col + ':')
        print(df[col].value_counts(dropna=False).to_string() + '\n')
    print('--- Verifications ---')
    nelements_check = (df['nelements'] == 3).all()
    print('Is nelements always 3? ' + str(nelements_check))
    if not nelements_check:
        print('Values of nelements that are not 3:')
        print(df.loc[df['nelements'] != 3, 'nelements'].value_counts().to_string())
    band_gap_check = (df['band_gap'] >= 0).all()
    print('Is band_gap >= 0 for all entries? ' + str(band_gap_check))
    if not band_gap_check:
        print('Values of band_gap < 0:')
        print(df.loc[df['band_gap'] < 0, 'band_gap'].to_string())
    if df['energy_above_hull'].isna().any():
        print('Warning: NaNs found in energy_above_hull. Ignoring NaNs for the check.')
    energy_hull_check = (df['energy_above_hull'].dropna() >= -0.01).all()
    print('Is energy_above_hull >= -0.01 eV/atom for all entries? ' + str(energy_hull_check))
    if not energy_hull_check:
        print('Values of energy_above_hull < -0.01:')
        print(df.loc[df['energy_above_hull'] < -0.01, 'energy_above_hull'].to_string())