# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

def run_pipeline():
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(data_path, index_col='material_id')
    uncharacterized_mask = df['is_elastic_characterized'] == False
    df_unchar = df[uncharacterized_mask].copy()
    print('========================================')
    print('--- PIPELINE INTEGRATION ---')
    print('========================================')
    print('Total uncharacterized materials: ' + str(len(df_unchar)))
    features = ['A_Z', 'B_Z', 'A_radius', 'B_radius', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'A_group', 'B_group', 'en_diff', 'tau', 'mu', 'B_valence', 'log_volume', 'abs_tau_diff', 'radius_diff', 'ie_ratio']
    X_base = df_unchar[features].copy()
    stability_model = joblib.load(os.path.join(data_dir, 'stability_model.joblib'))
    is_metal_clf = joblib.load(os.path.join(data_dir, 'is_metal_classifier.joblib'))
    band_gap_reg = joblib.load(os.path.join(data_dir, 'band_gap_regressor.joblib'))
    k_models = joblib.load(os.path.join(data_dir, 'k_vrh_quantiles_model.joblib'))
    g_models = joblib.load(os.path.join(data_dir, 'g_vrh_quantiles_model.joblib'))
    ductility_clf = joblib.load(os.path.join(data_dir, 'ductility_classifier.joblib'))
    pred_log_eah = stability_model.predict(X_base)
    pred_eah = np.expm1(pred_log_eah)
    pred_is_metal = is_metal_clf.predict(X_base)
    pred_band_gap = np.zeros(len(X_base))
    non_metal_mask = ~pred_is_metal.astype(bool)
    if non_metal_mask.sum() > 0:
        pred_band_gap[non_metal_mask] = band_gap_reg.predict(X_base[non_metal_mask])
    X_mech = X_base.copy()
    X_mech['pred_is_metal'] = pred_is_metal.astype(int)
    X_mech['pred_band_gap'] = pred_band_gap
    pred_k_50 = k_models['q50'].predict(X_mech)
    pred_k_05 = k_models['q05'].predict(X_mech)
    pred_k_95 = k_models['q95'].predict(X_mech)
    pred_g_50 = g_models['q50'].predict(X_mech)
    pred_g_05 = g_models['q05'].predict(X_mech)
    pred_g_95 = g_models['q95'].predict(X_mech)
    k_interval = pred_k_95 - pred_k_05
    g_interval = pred_g_95 - pred_g_05
    X_ductility = X_base.copy()
    X_ductility['pred_K_VRH'] = pred_k_50
    X_ductility['pred_G_VRH'] = pred_g_50
    pred_is_brittle = ductility_clf.predict(X_ductility)
    norm_stability = 1.0 - (pred_eah - pred_eah.min()) / (pred_eah.max() - pred_eah.min() + 1e-9)
    total_interval = k_interval + g_interval
    inv_interval = 1.0 / (total_interval + 1e-6)
    norm_confidence = (inv_interval - inv_interval.min()) / (inv_interval.max() - inv_interval.min() + 1e-9)
    ductility_bonus = 1.0 - pred_is_brittle
    raw_score = 0.4 * norm_stability + 0.4 * norm_confidence + 0.2 * ductility_bonus
    tau_filter = np.exp(-((X_base['tau'] - 0.85)**2) / (2 * 0.05**2))
    hp_score = raw_score * tau_filter
    results_df = pd.DataFrame({'formula': df_unchar['formula'] if 'formula' in df_unchar.columns else df_unchar.index, 'pred_energy_above_hull': pred_eah, 'pred_is_metal': pred_is_metal, 'pred_band_gap': pred_band_gap, 'pred_K_VRH_median': pred_k_50, 'pred_K_VRH_05': pred_k_05, 'pred_K_VRH_95': pred_k_95, 'pred_G_VRH_median': pred_g_50, 'pred_G_VRH_05': pred_g_05, 'pred_G_VRH_95': pred_g_95, 'pred_is_brittle': pred_is_brittle, 'tau': X_base['tau'], 'hp_score': hp_score}, index=df_unchar.index)
    results_df = results_df.sort_values(by='hp_score', ascending=False)
    print('\n========================================')
    print('--- PREDICTION SUMMARY ---')
    print('========================================')
    print('Predicted Stable (EAH < 0.05 eV/atom): ' + str((pred_eah < 0.05).sum()))
    print('Predicted Metals: ' + str(pred_is_metal.sum()))
    print('Predicted Ductile: ' + str((pred_is_brittle == 0).sum()))
    print('Predicted Brittle: ' + str((pred_is_brittle == 1).sum()))
    print('Median K_VRH: ' + str(round(np.median(pred_k_50), 2)) + ' GPa')
    print('Median G_VRH: ' + str(round(np.median(pred_g_50), 2)) + ' GPa')
    output_csv = os.path.join(data_dir, 'final_ranked_candidates.csv')
    results_df.to_csv(output_csv)
    print('\nFinal ranked candidate list saved to ' + output_csv)
    top_20 = results_df.head(20)
    print('\n========================================')
    print('--- TOP 20 HIGH-PERFORMANCE CANDIDATES ---')
    print('========================================')
    print(top_20[['formula', 'hp_score', 'pred_energy_above_hull', 'pred_K_VRH_median', 'pred_is_brittle']].to_string())
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ductile_mask = results_df['pred_is_brittle'] == 0
    brittle_mask = results_df['pred_is_brittle'] == 1
    axes[0].scatter(results_df.loc[ductile_mask, 'pred_K_VRH_median'], results_df.loc[ductile_mask, 'pred_energy_above_hull'], c='blue', label='Ductile', alpha=0.6, edgecolors='k')
    axes[0].scatter(results_df.loc[brittle_mask, 'pred_K_VRH_median'], results_df.loc[brittle_mask, 'pred_energy_above_hull'], c='red', label='Brittle', alpha=0.6, edgecolors='k')
    axes[0].set_xlabel('Predicted K_VRH (GPa)')
    axes[0].set_ylabel('Predicted Energy Above Hull (eV/atom)')
    axes[0].set_title('Stability vs. Bulk Modulus')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[1].hist(results_df['hp_score'], bins=30, color='purple', edgecolor='k', alpha=0.7)
    axes[1].set_xlabel('High-Performance Score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of HP Scores')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    y_pos = np.arange(len(top_20))
    axes[2].barh(y_pos, top_20['hp_score'][::-1], color='teal', edgecolor='k')
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(top_20['formula'][::-1])
    axes[2].set_xlabel('High-Performance Score')
    axes[2].set_title('Top 20 Candidates')
    axes[2].grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'pipeline_integration_1_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print('\nMulti-panel visualization saved to ' + plot_filename)

if __name__ == '__main__':
    run_pipeline()