# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(data_path, index_col='material_id')
    df['is_stable'] = df['energy_above_hull'] == 0
    categorical_cols = ['crystal_system', 'magnetic_ordering']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    encoded_cols = [col for col in df_encoded.columns if any(col.startswith(c + '_') for c in categorical_cols)]
    print('One-hot encoded column names:')
    for col in encoded_cols:
        print('- ' + col)
    X = df_encoded.drop(columns=['is_stable'])
    y = df_encoded['is_stable']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_counts = y_train.value_counts()
    train_pct = y_train.value_counts(normalize=True) * 100
    test_counts = y_test.value_counts()
    test_pct = y_test.value_counts(normalize=True) * 100
    print('\nClass distribution in Training Set:')
    print('True (Stable): ' + str(train_counts.get(True, 0)) + ' (' + str(round(train_pct.get(True, 0.0), 2)) + '%)')
    print('False (Unstable): ' + str(train_counts.get(False, 0)) + ' (' + str(round(train_pct.get(False, 0.0), 2)) + '%)')
    print('\nClass distribution in Test Set:')
    print('True (Stable): ' + str(test_counts.get(True, 0)) + ' (' + str(round(test_pct.get(True, 0.0), 2)) + '%)')
    print('False (Unstable): ' + str(test_counts.get(False, 0)) + ' (' + str(round(test_pct.get(False, 0.0), 2)) + '%)')
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(os.path.join(data_dir, 'train_dataset.csv'))
    test_df.to_csv(os.path.join(data_dir, 'test_dataset.csv'))
    print('\nSplits saved to data/train_dataset.csv and data/test_dataset.csv.')