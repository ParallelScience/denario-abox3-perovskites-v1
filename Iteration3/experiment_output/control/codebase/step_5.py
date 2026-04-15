# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import joblib
import time

plt.rcParams['text.usetex'] = False

def main():
    data_dir = 'data/'
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    df_all = pd.read_csv(os.path.join(data_dir, 'cleaned_perovskite_data.csv'))
    features = ['nsites', 'volume', 'density', 'density_atomic', 'spacegroup_number', 'A_Z', 'A_radius', 'A_en', 'A_ie1', 'A_group', 'B_Z', 'B_radius', 'B_en', 'B_ie1', 'B_group', 'B_valence', 'tau', 'mu', 'en_diff', 'VEC', 'volume_residual']
    X = df_all[features]
    y_metal = df_all['is_metal'].astype(int)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print('--- Electronic Profiling ---')
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=8)
    cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_metal_probs = cross_val_predict(rf_clf, X_scaled, y_metal, cv=cv_clf, method='predict_proba')[:, 1]
    roc_auc = roc_auc_score(y_metal, y_metal_probs)
    pr_auc = average_precision_score(y_metal, y_metal_probs)
    print('is_metal Classifier (5-Fold CV) - ROC-AUC: ' + str(round(roc_auc, 4)) + ', PR-AUC: ' + str(round(pr_auc, 4)))
    timestamp = str(int(time.time()))
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_metal, y_metal_probs)
    plt.plot(fpr, tpr, label='RF (AUC = ' + str(round(roc_auc, 4)) + ')')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - is_metal Classification')
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(data_dir, 'is_metal_roc_' + timestamp + '.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print('ROC curve saved to ' + roc_path)
    plt.figure(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_metal, y_metal_probs)
    plt.plot(rec, prec, label='RF (AP = ' + str(round(pr_auc, 4)) + ')')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve - is_metal Classification')
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(data_dir, 'is_metal_pr_' + timestamp + '.png')
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print('PR curve saved to ' + pr_path)
    rf_clf.fit(X_scaled, y_metal)
    non_metal_mask = df_all['is_metal'] == False
    X_nm = X_scaled[non_metal_mask]
    y_bg = df_all.loc[non_metal_mask, 'band_gap']
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    y_bg_preds = cross_val_predict(rf_reg, X_nm, y_bg, cv=cv_reg)
    r2_bg = r2_score(y_bg, y_bg_preds)
    rmse_bg = np.sqrt(mean_squared_error(y_bg, y_bg_preds))
    print('band_gap Regressor (Non-metals, 5-Fold CV) - R2: ' + str(round(r2_bg, 4)) + ', RMSE: ' + str(round(rmse_bg, 4)) + ' eV\n')
    rf_reg.fit(X_nm, y_bg)
    print('--- Uncertainty Propagation for Pugh Ratio ---')
    df_ranked = pd.read_csv(os.path.join(data_dir, 'ranked_candidates.csv'))
    df_in_domain = df_ranked[df_ranked['in_domain'] == True].copy()
    np.random.seed(42)
    n_samples = 10000
    pugh_means = []
    pugh_stds = []
    for _, row in df_in_domain.iterrows():
        K_samples = np.random.normal(row['K_VRH_pred'], row['K_VRH_std'], n_samples)
        G_samples = np.random.normal(row['G_VRH_pred'], row['G_VRH_std'], n_samples)
        valid_mask = K_samples > 1e-3
        if np.sum(valid_mask) > 0:
            p_samples = G_samples[valid_mask] / K_samples[valid_mask]
            pugh_means.append(np.mean(p_samples))
            pugh_stds.append(np.std(p_samples))
        else:
            pugh_means.append(np.nan)
            pugh_stds.append(np.nan)
    df_in_domain['pugh_ratio_pred'] = pugh_means
    df_in_domain['pugh_ratio_std'] = pugh_stds
    print('Monte Carlo sampling completed for ' + str(len(df_in_domain)) + ' In-Domain materials.\n')
    print('--- Multi-Objective Optimization ---')
    X_in_domain_raw = df_in_domain[features]
    X_in_domain_scaled = scaler.transform(imputer.transform(X_in_domain_raw))
    pred_is_metal = rf_clf.predict(X_in_domain_scaled)
    pred_bg_raw = rf_reg.predict(X_in_domain_scaled)
    df_in_domain['pred_band_gap'] = np.where(pred_is_metal == 1, 0.0, pred_bg_raw)
    objs = df_in_domain[['prob_stable', 'G_VRH_pred', 'pred_band_gap']].values
    is_pareto = np.ones(objs.shape[0], dtype=bool)
    for i, c in enumerate(objs):
        diff = objs - c
        dominated_by = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
        if np.any(dominated_by):
            is_pareto[i] = False
    df_pareto = df_in_domain[is_pareto].copy()
    print('Identified ' + str(len(df_pareto)) + ' Pareto-optimal candidates out of ' + str(len(df_in_domain)) + ' In-Domain materials.')
    df_pareto = df_pareto.merge(df_all[['material_id', 'A_site', 'B_site', 'formula']], on='material_id', how='left', suffixes=('', '_y'))
    if 'formula_y' in df_pareto.columns:
        df_pareto.drop(columns=['formula_y'], inplace=True)
    unique_A = df_pareto['A_site'].nunique()
    unique_B = df_pareto['B_site'].nunique()
    print('Chemical Diversity of Pareto Candidates: ' + str(unique_A) + ' unique A-site elements, ' + str(unique_B) + ' unique B-site elements.\n')
    print('--- Generating Plots ---')
    elastic_df = df_all.dropna(subset=['K_VRH', 'G_VRH']).copy()
    elastic_df = elastic_df[(elastic_df['K_VRH'] <= 300) & (elastic_df['G_VRH'] >= 0) & (elastic_df['G_VRH'] <= 200)]
    X_el = elastic_df[features]
    y_K = elastic_df['K_VRH']
    y_G = elastic_df['G_VRH']
    gpr_imputer = joblib.load(os.path.join(data_dir, 'gpr_imputer.joblib'))
    gpr_scaler = joblib.load(os.path.join(data_dir, 'gpr_scaler.joblib'))
    X_el_scaled = gpr_scaler.transform(gpr_imputer.transform(X_el))
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    gpr_cv = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=True)
    y_K_cv = cross_val_predict(gpr_cv, X_el_scaled, y_K, cv=5)
    y_G_cv = cross_val_predict(gpr_cv, X_el_scaled, y_G, cv=5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_K, y_K_cv, alpha=0.6, edgecolors='k')
    axes[0].plot([0, 300], [0, 300], 'r--')
    axes[0].set_xlabel('Actual K_VRH (GPa)')
    axes[0].set_ylabel('Predicted K_VRH (GPa)')
    axes[0].set_title('GPR Parity Plot - Bulk Modulus')
    axes[1].scatter(y_G, y_G_cv, alpha=0.6, edgecolors='k')
    axes[1].plot([0, 200], [0, 200], 'r--')
    axes[1].set_xlabel('Actual G_VRH (GPa)')
    axes[1].set_ylabel('Predicted G_VRH (GPa)')
    axes[1].set_title('GPR Parity Plot - Shear Modulus')
    plt.tight_layout()
    parity_path = os.path.join(data_dir, 'gpr_parity_' + timestamp + '.png')
    plt.savefig(parity_path, dpi=300)
    plt.close()
    print('GPR parity plot saved to ' + parity_path)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df_in_domain['prob_stable'], df_in_domain['G_VRH_pred'], c=df_in_domain['pred_band_gap'], cmap='viridis', alpha=0.5, label='In-Domain')
    plt.scatter(df_pareto['prob_stable'], df_pareto['G_VRH_pred'], facecolors='none', edgecolors='r', s=100, linewidths=1.5, label='Pareto Optimal')
    plt.colorbar(sc, label='Predicted Band Gap (eV)')
    plt.xlabel('Predicted Stability Probability')
    plt.ylabel('Predicted G_VRH (GPa)')
    plt.title('Multi-Objective Space (In-Domain Materials)')
    plt.legend()
    plt.tight_layout()
    pareto_path = os.path.join(data_dir, 'pareto_frontier_' + timestamp + '.png')
    plt.savefig(pareto_path, dpi=300)
    plt.close()
    print('Pareto frontier plot saved to ' + pareto_path)
    pca = PCA(n_components=2, random_state=42)
    X_all_pca = pca.fit_transform(X_scaled)
    df_all['PCA1'] = X_all_pca[:, 0]
    df_all['PCA2'] = X_all_pca[:, 1]
    elastic_mask = df_all['K_VRH'].notnull() & df_all['G_VRH'].notnull()
    df_char = df_all[elastic_mask]
    df_unchar = df_all[~elastic_mask].copy()
    df_unchar = df_unchar.merge(df_ranked[['material_id', 'domain_status']], on='material_id', how='left')
    plt.figure(figsize=(8, 6))
    plt.scatter(df_unchar[df_unchar['domain_status'] == 'Out-of-Domain (High-Uncertainty)']['PCA1'], df_unchar[df_unchar['domain_status'] == 'Out-of-Domain (High-Uncertainty)']['PCA2'], alpha=0.4, label='Out-of-Domain', color='red', s=20)
    plt.scatter(df_unchar[df_unchar['domain_status'] == 'In-Domain']['PCA1'], df_unchar[df_unchar['domain_status'] == 'In-Domain']['PCA2'], alpha=0.4, label='In-Domain', color='blue', s=20)
    plt.scatter(df_char['PCA1'], df_char['PCA2'], alpha=0.8, label='Characterized (Training)', color='green', marker='^', s=40, edgecolors='k')
    plt.xlabel('PCA 1 (' + str(round(pca.explained_variance_ratio_[0]*100, 1)) + '%)')
    plt.ylabel('PCA 2 (' + str(round(pca.explained_variance_ratio_[1]*100, 1)) + '%)')
    plt.title('Applicability Domain PCA Projection')
    plt.legend()
    plt.tight_layout()
    pca_path = os.path.join(data_dir, 'domain_pca_' + timestamp + '.png')
    plt.savefig(pca_path, dpi=300)
    plt.close()
    print('PCA projection plot saved to ' + pca_path)
    plt.figure(figsize=(8, 6))
    plt.hist(df_ranked['UCB_score'], bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('UCB Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of UCB Scores for Uncharacterized Materials')
    plt.tight_layout()
    ucb_path = os.path.join(data_dir, 'ucb_distribution_' + timestamp + '.png')
    plt.savefig(ucb_path, dpi=300)
    plt.close()
    print('UCB distribution plot saved to ' + ucb_path)
    pareto_out_path = os.path.join(data_dir, 'pareto_candidates.csv')
    df_pareto.to_csv(pareto_out_path, index=False)
    print('\nPareto candidates saved to ' + pareto_out_path)
    print('\n--- Top 10 Pareto Candidates (Sorted by Stability Probability) ---')
    cols_to_print = ['material_id', 'formula', 'prob_stable', 'G_VRH_pred', 'pred_band_gap', 'pugh_ratio_pred', 'pugh_ratio_std']
    print(df_pareto.sort_values('prob_stable', ascending=False)[cols_to_print].head(10).to_string(index=False))

if __name__ == '__main__':
    main()