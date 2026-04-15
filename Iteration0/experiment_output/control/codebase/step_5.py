# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
import joblib

if __name__ == '__main__':
    data_dir = 'data/'
    full_data_path = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    full_df = pd.read_csv(full_data_path, index_col='material_id')
    categorical_cols = ['crystal_system', 'magnetic_ordering']
    full_df_encoded = pd.get_dummies(full_df, columns=categorical_cols, drop_first=False, dtype=int)
    filtered_path = os.path.join(data_dir, 'filtered_elastic_dataset.csv')
    filtered_df = pd.read_csv(filtered_path, index_col='material_id')
    df_encoded = full_df_encoded.loc[filtered_df.index].copy()
    leakage_cols = ['energy_above_hull', 'formation_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'energy_per_atom', 'is_stable']
    mechanical_cols = ['K_VRH', 'K_voigt', 'K_reuss', 'G_VRH', 'G_voigt', 'G_reuss', 'elastic_anisotropy', 'poisson_ratio', 'pugh_ratio']
    string_cols = ['formula', 'chemsys', 'spacegroup_symbol', 'A_site', 'B_site']
    cols_to_drop = leakage_cols + mechanical_cols + string_cols
    X = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number, bool]).astype(float)
    y_K = df_encoded['K_VRH']
    y_G = df_encoded['G_VRH']
    rf_K = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=8)
    rf_G = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=8)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_K = cross_validate(rf_K, X, y_K, cv=cv, scoring=('r2', 'neg_mean_absolute_error'), return_estimator=False)
    cv_G = cross_validate(rf_G, X, y_G, cv=cv, scoring=('r2', 'neg_mean_absolute_error'), return_estimator=False)
    print('K_VRH Regressor Cross-Validation Results:')
    print('R2: ' + str(round(np.mean(cv_K['test_r2']), 4)) + ' ± ' + str(round(np.std(cv_K['test_r2']), 4)))
    print('MAE: ' + str(round(-np.mean(cv_K['test_neg_mean_absolute_error']), 4)) + ' ± ' + str(round(np.std(cv_K['test_neg_mean_absolute_error']), 4)))
    print('\nG_VRH Regressor Cross-Validation Results:')
    print('R2: ' + str(round(np.mean(cv_G['test_r2']), 4)) + ' ± ' + str(round(np.std(cv_G['test_r2']), 4)))
    print('MAE: ' + str(round(-np.mean(cv_G['test_neg_mean_absolute_error']), 4)) + ' ± ' + str(round(np.std(cv_G['test_neg_mean_absolute_error']), 4)))
    rf_K.fit(X, y_K)
    rf_G.fit(X, y_G)
    importances_K = rf_K.feature_importances_
    importances_G = rf_G.feature_importances_
    df_imp_K = pd.DataFrame({'Feature': X.columns, 'Importance': importances_K}).sort_values(by='Importance', ascending=False)
    df_imp_G = pd.DataFrame({'Feature': X.columns, 'Importance': importances_G}).sort_values(by='Importance', ascending=False)
    print('\nTop 10 Features for K_VRH:')
    print(df_imp_K.head(10).to_string(index=False))
    print('\nTop 10 Features for G_VRH:')
    print(df_imp_G.head(10).to_string(index=False))
    joblib.dump(rf_K, os.path.join(data_dir, 'rf_K_VRH_model.joblib'))
    joblib.dump(rf_G, os.path.join(data_dir, 'rf_G_VRH_model.joblib'))
    joblib.dump(list(X.columns), os.path.join(data_dir, 'mechanical_features.joblib'))
    print('\nModels saved to data/rf_K_VRH_model.joblib and data/rf_G_VRH_model.joblib')
    print('Feature list saved to data/mechanical_features.joblib')