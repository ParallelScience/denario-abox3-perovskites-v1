# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import precision_recall_curve, average_precision_score, mean_absolute_error, r2_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

def main():
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'cleaned_dataset.csv')
    df = pd.read_csv(data_path)
    if 'material_id' in df.columns:
        df = df.set_index('material_id')
    exclude_cols = ['energy_above_hull', 'formation_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'is_stable', 'band_gap', 'is_gap_direct', 'is_metal', 'efermi', 'is_magnetic', 'total_magnetization', 'total_magnetization_per_fu', 'num_magnetic_sites', 'K_VRH', 'K_voigt', 'K_reuss', 'G_VRH', 'G_voigt', 'G_reuss', 'elastic_anisotropy', 'poisson_ratio', 'pugh_ratio', 'energy_per_atom', 'formula', 'chemsys', 'spacegroup_symbol']
    mag_cols = [col for col in df.columns if col.startswith('magnetic_ordering_')]
    exclude_cols.extend(mag_cols)
    X_all = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    X_all = X_all.select_dtypes(include=[np.number])
    X_all = X_all.fillna(X_all.median())
    y_metal = df['is_metal'].astype(int)
    skf_metal = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gbc_metal = GradientBoostingClassifier(n_estimators=100, random_state=42)
    oof_preds_metal = np.zeros(len(X_all))
    for train_idx, test_idx in skf_metal.split(X_all, y_metal):
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_metal.iloc[train_idx], y_metal.iloc[test_idx]
        gbc_metal.fit(X_train, y_train)
        oof_preds_metal[test_idx] = gbc_metal.predict_proba(X_test)[:, 1]
    gbc_metal.fit(X_all, y_metal)
    non_metal_idx = df[df['is_metal'] == False].index
    X_nm = X_all.loc[non_metal_idx]
    y_bg = df.loc[non_metal_idx, 'band_gap']
    kf_bg = KFold(n_splits=5, shuffle=True, random_state=42)
    gbr_bg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr_bg.fit(X_nm, y_bg)
    df_elastic = df.dropna(subset=['K_VRH', 'G_VRH']).copy()
    k_p1, k_p99 = df_elastic['K_VRH'].quantile(0.01), df_elastic['K_VRH'].quantile(0.99)
    g_p1, g_p99 = df_elastic['G_VRH'].quantile(0.01), df_elastic['G_VRH'].quantile(0.99)
    viable_mask = ((df_elastic['K_VRH'] >= k_p1) & (df_elastic['K_VRH'] <= k_p99) & (df_elastic['G_VRH'] >= g_p1) & (df_elastic['G_VRH'] <= g_p99))
    df_viable = df_elastic[viable_mask].copy()
    pugh = df_viable['pugh_ratio'] if 'pugh_ratio' in df_viable.columns and not df_viable['pugh_ratio'].isnull().all() else df_viable['G_VRH'] / df_viable['K_VRH']
    df_viable['is_ductile'] = (pugh < 0.571).astype(int)
    y_ductile = df_viable['is_ductile']
    X_ductile = X_all.loc[df_viable.index]
    gbc_ductile = GradientBoostingClassifier(n_estimators=100, random_state=42)
    weights = compute_sample_weight(class_weight='balanced', y=y_ductile)
    gbc_ductile.fit(X_ductile, y_ductile, sample_weight=weights)
    df['pred_is_metal'] = gbc_metal.predict_proba(X_all)[:, 1]
    df['pred_band_gap'] = gbr_bg.predict(X_all)
    df.loc[df['pred_is_metal'] > 0.5, 'pred_band_gap'] = 0.0
    df['pred_is_ductile'] = gbc_ductile.predict(X_all)
    df.to_csv(os.path.join(data_dir, 'electronic_ductility_predictions.csv'))
    print('Saved to data/electronic_ductility_predictions.csv')

if __name__ == '__main__':
    main()