# filename: codebase/step_2.py
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
    cols_to_drop = [col for col in df.columns if col.endswith('.1')]
    df = df.drop(columns=cols_to_drop)
    print('='*50)
    print('--- Summary Statistics for Numeric Columns ---')
    print('='*50)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary_stats = df[numeric_cols].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99])
    print(summary_stats.to_string() + '\n')
    print('='*50)
    print('--- Value Counts and Percentages for Boolean and Categorical Columns ---')
    print('='*50)
    bool_cols = ['theoretical', 'is_stable', 'is_metal', 'is_gap_direct', 'is_magnetic']
    cat_cols = ['crystal_system', 'magnetic_ordering', 'spacegroup_symbol']
    for col in bool_cols + cat_cols:
        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(dropna=False, normalize=True) * 100
        stats_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
        print('Column: ' + col)
        print(stats_df.to_string() + '\n')
    print('='*50)
    print('--- Outlier Detection (IQR Method) ---')
    print('='*50)
    outlier_cols = ['volume', 'band_gap', 'K_VRH', 'G_VRH', 'formation_energy_per_atom', 'energy_above_hull', 'tau', 'mu']
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            non_null_count = df[col].notna().sum()
            print('Outliers for ' + col + ':')
            print('Q1: ' + str(Q1) + ', Q3: ' + str(Q3) + ', IQR: ' + str(IQR))
            print('Lower Bound: ' + str(lower_bound) + ', Upper Bound: ' + str(upper_bound))
            print('Number of outliers: ' + str(len(outliers)) + ' (' + str(round(len(outliers)/non_null_count*100, 2)) + '% of non-null values)\n')
    print('='*50)
    print('--- Correlation Matrix ---')
    print('='*50)
    corr_cols = ['volume', 'density', 'formation_energy_per_atom', 'energy_above_hull', 'tau', 'mu', 'A_en', 'B_en', 'A_radius', 'B_radius', 'en_diff', 'radius_ratio', 'band_gap', 'K_VRH', 'G_VRH']
    corr_matrix = df[corr_cols].corr()
    print('Correlation matrix for selected physically meaningful features:')
    print(corr_matrix.to_string() + '\n')
    print('Top absolute correlations (excluding self-correlations):')
    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    def sort_index(idx):
        return tuple(sorted(idx))
    corr_pairs.index = pd.MultiIndex.from_tuples([sort_index(idx) for idx in corr_pairs.index])
    corr_pairs = corr_pairs[~corr_pairs.index.duplicated()]
    corr_pairs_sorted = corr_pairs.sort_values(key=abs, ascending=False)
    print(corr_pairs_sorted.head(20).to_string() + '\n')