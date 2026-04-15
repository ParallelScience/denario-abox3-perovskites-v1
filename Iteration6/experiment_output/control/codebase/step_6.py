# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
import shap

def run_interpretability_and_optimization():
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_perovskite_data.csv'))
    stability_model = joblib.load(os.path.join(data_dir, 'stability_gbc_model.joblib'))
    stability_features = joblib.load(os.path.join(data_dir, 'stability_gbc_features.joblib'))
    viability_model = joblib.load(os.path.join(data_dir, 'mechanical_viability_classifier.joblib'))
    viability_features = joblib.load(os.path.join(data_dir, 'mechanical_viability_features.joblib'))
    gpr_K = joblib.load(os.path.join(data_dir, 'gpr_K_model.joblib'))
    gpr_G = joblib.load(os.path.join(data_dir, 'gpr_G_model.joblib'))
    mech_features = joblib.load(os.path.join(data_dir, 'mechanical_features.joblib'))
    X_stab = df[['tau_strain', 'mu_strain', 'tau', 'mu', 'en_diff', 'glazer_tilt']].copy()
    X_stab = pd.get_dummies(X_stab, columns=['glazer_tilt'], drop_first=False).astype(float)
    for col in stability_features:
        if col not in X_stab.columns:
            X_stab[col] = 0.0
    X_stab = X_stab[stability_features]
    base_features_viab = ['tau_strain', 'mu_strain', 'tau', 'mu', 'en_diff', 'A_radius', 'B_radius', 'density', 'formation_energy_per_atom', 'log_volume']
    crystal_dummies = pd.get_dummies(df['crystal_system'], prefix='crystal', drop_first=False).astype(float)
    glazer_dummies = pd.get_dummies(df['glazer_tilt'], prefix='tilt', drop_first=False).astype(float)
    X_viab = pd.concat([df[base_features_viab], crystal_dummies, glazer_dummies], axis=1)
    for col in viability_features:
        if col not in X_viab.columns:
            X_viab[col] = 0.0
    X_viab = X_viab[viability_features]
    base_features_mech = ['density', 'tau', 'mu', 'en_diff', 'A_radius', 'B_radius', 'formation_energy_per_atom']
    X_mech = pd.concat([df[base_features_mech], crystal_dummies], axis=1)
    for col in mech_features:
        if col not in X_mech.columns:
            X_mech[col] = 0.0
    X_mech = X_mech[mech_features]
    print('Running SHAP analysis for Stability Model...')
    explainer_stab = shap.TreeExplainer(stability_model)
    shap_values_stab = explainer_stab.shap_values(X_stab)
    if isinstance(shap_values_stab, list):
        shap_values_stab = shap_values_stab[1]
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_stab, X_stab, show=False)
    plt.title('SHAP Summary: Thermodynamic Stability')
    plt.tight_layout()
    timestamp = int(time.time())
    stab_shap_path = os.path.join(data_dir, 'shap_summary_stability_1_' + str(timestamp) + '.png')
    plt.savefig(stab_shap_path, dpi=300)
    plt.close()
    print('Stability SHAP plot saved to ' + stab_shap_path)
    print('Running SHAP analysis for Mechanical Viability Model...')
    explainer_viab = shap.TreeExplainer(viability_model)
    shap_values_viab = explainer_viab.shap_values(X_viab)
    if isinstance(shap_values_viab, list):
        shap_values_viab = shap_values_viab[1]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_viab, X_viab, show=False)
    plt.title('SHAP Summary: Mechanical Viability')
    plt.tight_layout()
    viab_shap_path = os.path.join(data_dir, 'shap_summary_viability_1_' + str(timestamp) + '.png')
    plt.savefig(viab_shap_path, dpi=300)
    plt.close()
    print('Mechanical Viability SHAP plot saved to ' + viab_shap_path)
    print('\nGenerating predictions for all materials...')
    df['stability_prob'] = stability_model.predict_proba(X_stab)[:, 1]
    df['viability_prob'] = viability_model.predict_proba(X_viab)[:, 1]
    _, K_std = gpr_K.predict(X_mech, return_std=True)
    _, G_std = gpr_G.predict(X_mech, return_std=True)
    df['K_var'] = K_std ** 2
    df['G_var'] = G_std ** 2
    norm_K_var = df['K_var'] / df['K_var'].max()
    norm_G_var = df['G_var'] / df['G_var'].max()
    df['gpr_uncertainty_penalty'] = (norm_K_var + norm_G_var) / 2.0
    df['penalized_viability_score'] = df['viability_prob'] * (1.0 - df['gpr_uncertainty_penalty'])
    uncharacterized_mask = df['K_VRH'].isnull()
    candidates = df[uncharacterized_mask].copy()
    print('Total uncharacterized candidates: ' + str(len(candidates)))
    points = candidates[['stability_prob', 'penalized_viability_score']].values
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        dominating = (points >= c).all(axis=1) & (points > c).any(axis=1)
        if dominating.any():
            is_pareto[i] = False
    pareto_candidates = candidates[is_pareto].copy()
    print('Found ' + str(len(pareto_candidates)) + ' Pareto-optimal candidates.')
    pareto_candidates['combined_score'] = pareto_candidates['stability_prob'] + pareto_candidates['penalized_viability_score']
    top_20 = pareto_candidates.sort_values(by='combined_score', ascending=False).head(20)
    cols_to_save = ['formula', 'material_id', 'stability_prob', 'viability_prob', 'penalized_viability_score', 'gpr_uncertainty_penalty', 'K_var', 'G_var', 'tau', 'mu', 'en_diff', 'density', 'formation_energy_per_atom']
    top_20_save = top_20[cols_to_save]
    top_20_path = os.path.join(data_dir, 'top_20_pareto_candidates.csv')
    top_20_save.to_csv(top_20_path, index=False)
    print('Top 20 Pareto candidates saved to ' + top_20_path)
    print('\n--- Top 5 Pareto Candidates ---')
    print(top_20_save.head(5).to_string(index=False))
    plt.figure(figsize=(10, 8))
    plt.scatter(candidates['stability_prob'], candidates['penalized_viability_score'], alpha=0.5, c='gray', label='Uncharacterized Candidates')
    plt.scatter(pareto_candidates['stability_prob'], pareto_candidates['penalized_viability_score'], c='red', label='Pareto Front', edgecolor='k', s=50)
    plt.scatter(top_20['stability_prob'], top_20['penalized_viability_score'], c='gold', marker='*', s=200, edgecolor='k', label='Top 20 Candidates')
    plt.xlabel('Thermodynamic Stability Probability')
    plt.ylabel('Penalized Mechanical Viability Score')
    plt.title('Trade-off Space: Stability vs. Mechanical Robustness')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    tradeoff_plot_path = os.path.join(data_dir, 'tradeoff_space_1_' + str(timestamp) + '.png')
    plt.savefig(tradeoff_plot_path, dpi=300)
    plt.close()
    print('Trade-off space plot saved to ' + tradeoff_plot_path)

if __name__ == '__main__':
    run_interpretability_and_optimization()