# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

def train_mechanical_models():
    data_path = 'data/cleaned_perovskite_data.csv'
    df = pd.read_csv(data_path, index_col='material_id')
    elastic_mask = df['is_elastic_characterized'] == True
    df_elastic = df[elastic_mask].copy()
    valid_k = (df_elastic['K_VRH'] > 0) & (df_elastic['K_VRH'] < 300)
    valid_g = (df_elastic['G_VRH'] > 0) & (df_elastic['G_VRH'] < 200)
    df_filtered = df_elastic[valid_k & valid_g].copy()
    print('========================================')
    print('--- MECHANICAL PROPERTY FILTERING ---')
    print('========================================')
    print('Original elastic subset size: ' + str(len(df_elastic)))
    print('Samples retained after filtering (0 < K < 300, 0 < G < 200): ' + str(len(df_filtered)))
    features = ['A_Z', 'B_Z', 'A_radius', 'B_radius', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'A_group', 'B_group', 'en_diff', 'tau', 'mu', 'B_valence', 'log_volume', 'abs_tau_diff', 'radius_diff', 'ie_ratio']
    X_base = df_filtered[features]
    clf = joblib.load('data/is_metal_classifier.joblib')
    reg = joblib.load('data/band_gap_regressor.joblib')
    pred_is_metal = clf.predict(X_base)
    pred_band_gap = np.zeros(len(X_base))
    non_metal_mask = ~pred_is_metal.astype(bool)
    if non_metal_mask.sum() > 0:
        pred_band_gap[non_metal_mask] = reg.predict(X_base[non_metal_mask])
    X_mech = X_base.copy()
    X_mech['pred_is_metal'] = pred_is_metal.astype(int)
    X_mech['pred_band_gap'] = pred_band_gap
    y_k = df_filtered['K_VRH']
    y_g = df_filtered['G_VRH']
    gbr_k_50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=42)
    gbr_k_05 = GradientBoostingRegressor(loss='quantile', alpha=0.05, random_state=42)
    gbr_k_95 = GradientBoostingRegressor(loss='quantile', alpha=0.95, random_state=42)
    gbr_g_50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=42)
    gbr_g_05 = GradientBoostingRegressor(loss='quantile', alpha=0.05, random_state=42)
    gbr_g_95 = GradientBoostingRegressor(loss='quantile', alpha=0.95, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    def get_cv_predictions(model, X, y):
        preds = np.zeros(len(y))
        for train_idx, test_idx in kf.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds[test_idx] = model.predict(X.iloc[test_idx])
        return preds
    k_50_pred = get_cv_predictions(gbr_k_50, X_mech, y_k)
    k_05_pred = get_cv_predictions(gbr_k_05, X_mech, y_k)
    k_95_pred = get_cv_predictions(gbr_k_95, X_mech, y_k)
    g_50_pred = get_cv_predictions(gbr_g_50, X_mech, y_g)
    g_05_pred = get_cv_predictions(gbr_g_05, X_mech, y_g)
    g_95_pred = get_cv_predictions(gbr_g_95, X_mech, y_g)
    k_interval_width = k_95_pred - k_05_pred
    g_interval_width = g_95_pred - g_05_pred
    print('\n========================================')
    print('--- QUANTILE REGRESSION METRICS ---')
    print('========================================')
    print('Median prediction interval width (90% CI) for K_VRH: ' + str(round(np.median(k_interval_width), 4)) + ' GPa')
    print('Median prediction interval width (90% CI) for G_VRH: ' + str(round(np.median(g_interval_width), 4)) + ' GPa')
    gbr_k_50.fit(X_mech, y_k)
    gbr_k_05.fit(X_mech, y_k)
    gbr_k_95.fit(X_mech, y_k)
    gbr_g_50.fit(X_mech, y_g)
    gbr_g_05.fit(X_mech, y_g)
    gbr_g_95.fit(X_mech, y_g)
    joblib.dump({'q05': gbr_k_05, 'q50': gbr_k_50, 'q95': gbr_k_95}, 'data/k_vrh_quantiles_model.joblib')
    joblib.dump({'q05': gbr_g_05, 'q50': gbr_g_50, 'q95': gbr_g_95}, 'data/g_vrh_quantiles_model.joblib')
    print('\nModels saved to data/k_vrh_quantiles_model.joblib and data/g_vrh_quantiles_model.joblib')
    results_df = pd.DataFrame({'K_VRH_actual': y_k, 'K_VRH_pred_50': k_50_pred, 'K_VRH_pred_05': k_05_pred, 'K_VRH_pred_95': k_95_pred, 'G_VRH_actual': y_g, 'G_VRH_pred_50': g_50_pred, 'G_VRH_pred_05': g_05_pred, 'G_VRH_pred_95': g_95_pred}, index=X_mech.index)
    results_df.to_csv('data/mechanical_cv_predictions.csv')
    print('Cross-validated predictions saved to data/mechanical_cv_predictions.csv')
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    k_err_lower = np.maximum(0, k_50_pred - k_05_pred)
    k_err_upper = np.maximum(0, k_95_pred - k_50_pred)
    g_err_lower = np.maximum(0, g_50_pred - g_05_pred)
    g_err_upper = np.maximum(0, g_95_pred - g_50_pred)
    axes[0].errorbar(y_k, k_50_pred, yerr=[k_err_lower, k_err_upper], fmt='o', alpha=0.6, ecolor='lightgray', capsize=3, markerfacecolor='blue', markeredgecolor='k')
    axes[0].plot([y_k.min(), y_k.max()], [y_k.min(), y_k.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual K_VRH (GPa)')
    axes[0].set_ylabel('Predicted K_VRH (Median, GPa)')
    axes[0].set_title('K_VRH: Predicted vs Actual (90% CI)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[1].errorbar(y_g, g_50_pred, yerr=[g_err_lower, g_err_upper], fmt='o', alpha=0.6, ecolor='lightgray', capsize=3, markerfacecolor='green', markeredgecolor='k')
    axes[1].plot([y_g.min(), y_g.max()], [y_g.min(), y_g.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual G_VRH (GPa)')
    axes[1].set_ylabel('Predicted G_VRH (Median, GPa)')
    axes[1].set_title('G_VRH: Predicted vs Actual (90% CI)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join('data', 'mechanical_quantiles_1_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print('Plot saved to ' + plot_filename)

if __name__ == '__main__':
    train_mechanical_models()