# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
from sklearn.inspection import PartialDependenceDisplay
import time

def run_validation():
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(data_path, index_col='material_id')
    features = ['A_Z', 'B_Z', 'A_radius', 'B_radius', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'A_group', 'B_group', 'en_diff', 'tau', 'mu', 'B_valence', 'log_volume', 'abs_tau_diff', 'radius_diff', 'ie_ratio']
    X = df[features].values
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = pinv(cov)
    distances = np.array([mahalanobis(x, mu, inv_cov) for x in X])
    threshold = np.percentile(distances, 97.5)
    is_ood = distances > threshold
    df['mahalanobis_distance'] = distances
    df['is_ood'] = is_ood
    ood_count = is_ood.sum()
    print('========================================')
    print('--- OUT-OF-DISTRIBUTION (OOD) DETECTION ---')
    print('========================================')
    print('97.5th Percentile Threshold: ' + str(round(threshold, 4)))
    print('Number of OOD materials flagged: ' + str(ood_count))
    val_results_path = os.path.join(data_dir, 'validation_results.csv')
    df[['mahalanobis_distance', 'is_ood']].to_csv(val_results_path)
    print('Validation results saved to ' + val_results_path)
    stability_model = joblib.load(os.path.join(data_dir, 'stability_model.joblib'))
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].hist(distances, bins=40, color='skyblue', edgecolor='k', alpha=0.7)
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label='97.5th Percentile')
    axes[0].set_xlabel('Mahalanobis Distance')
    axes[0].set_ylabel('Count')
    axes[0].set_title('OOD Detection: Mahalanobis Distances')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    PartialDependenceDisplay.from_estimator(stability_model, df[features], ['tau'], ax=axes[1])
    axes[1].axvspan(0.8, 1.0, color='green', alpha=0.2, label='Ideal Perovskite Range (0.8-1.0)')
    axes[1].set_title('PDP: Stability vs. Tolerance Factor (tau)')
    axes[1].set_ylabel('Partial Dependence (log1p EAH)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    elastic_mask = df['is_elastic_characterized'] == True
    df_elastic = df[elastic_mask]
    valid_k = (df_elastic['K_VRH'] > 0) & (df_elastic['K_VRH'] < 300)
    df_elastic_filtered = df_elastic[valid_k]
    preds_path = os.path.join(data_dir, 'final_ranked_candidates.csv')
    if os.path.exists(preds_path):
        df_preds = pd.read_csv(preds_path, index_col='material_id')
        df_preds = df_preds.join(df[['log_volume']])
        corr = df_preds['log_volume'].corr(df_preds['pred_K_VRH_median'])
        print('\n========================================')
        print('--- PHYSICAL CONSISTENCY CHECK ---')
        print('========================================')
        print('Correlation between log_volume and predicted K_VRH: ' + str(round(corr, 4)))
        axes[2].scatter(df_preds['log_volume'], df_preds['pred_K_VRH_median'], alpha=0.5, color='orange', edgecolors='k', label='Predicted (Uncharacterized)')
    axes[2].scatter(df_elastic_filtered['log_volume'], df_elastic_filtered['K_VRH'], alpha=0.7, color='blue', edgecolors='k', label='Actual (Filtered)')
    axes[2].set_xlabel('Log Volume (log(1 + V))')
    axes[2].set_ylabel('Bulk Modulus K_VRH (GPa)')
    axes[2].set_title('Physical Consistency: K_VRH vs. Log Volume')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'validation_checks_1_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print('\nValidation plots saved to ' + plot_filename)

if __name__ == '__main__':
    run_validation()