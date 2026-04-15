# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import os

if __name__ == '__main__':
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(data_path, index_col='material_id')
    elastic_df = df.dropna(subset=['K_VRH', 'G_VRH'])
    mask = (elastic_df['K_VRH'] > 0) & (elastic_df['K_VRH'] < 300) & (elastic_df['G_VRH'] > 0) & (elastic_df['G_VRH'] < 200)
    filtered_elastic_df = elastic_df[mask].copy()
    print('Number of samples retained after filtering: ' + str(len(filtered_elastic_df)))
    print('\nDescriptive statistics for filtered K_VRH and G_VRH (in GPa):')
    stats = filtered_elastic_df[['K_VRH', 'G_VRH']].agg(['mean', 'std', 'min', 'max'])
    print(stats.to_string())
    output_path = os.path.join(data_dir, 'filtered_elastic_dataset.csv')
    filtered_elastic_df.to_csv(output_path)
    print('\nFiltered elastic subset saved to ' + output_path)