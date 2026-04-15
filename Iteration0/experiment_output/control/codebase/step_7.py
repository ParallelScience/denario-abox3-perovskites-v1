# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    feat_imp_df = pd.read_csv(os.path.join(data_dir, 'stability_feature_importances.csv'))
    preds_df = pd.read_csv(os.path.join(data_dir, 'pipeline_predictions.csv'), index_col='material_id')
    full_df = pd.read_csv(os.path.join(data_dir, 'cleaned_perovskite_data.csv'), index_col='material_id')
    preds_df = preds_df.join(full_df[['tau']])
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    top_10_imp = feat_imp_df.head(10)
    sns.barplot(x='Importance', y='Feature', data=top_10_imp, ax=axs[0, 0], palette='viridis', hue='Feature', legend=False)
    axs[0, 0].set_title('(a) Top 10 Feature Importances (Stability Classifier)')
    axs[0, 0].set_xlabel('Importance Score')
    axs[0, 0].set_ylabel('Feature')
    sc = axs[0, 1].scatter(preds_df['K_VRH_pred'], preds_df['G_VRH_pred'], c=preds_df['stability_prob'], cmap='coolwarm', alpha=0.7, edgecolor='k')
    axs[0, 1].set_title('(b) Predicted K_VRH vs G_VRH')
    axs[0, 1].set_xlabel('Predicted K_VRH (GPa)')
    axs[0, 1].set_ylabel('Predicted G_VRH (GPa)')
    cbar = fig.colorbar(sc, ax=axs[0, 1])
    cbar.set_label('Stability Probability')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].scatter(preds_df['tau'], preds_df['stability_prob'], alpha=0.5, color='teal', edgecolor='k')
    axs[1, 0].set_title('(c) Tolerance Factor (tau) vs Stability Probability')
    axs[1, 0].set_xlabel('Goldschmidt Tolerance Factor (tau)')
    axs[1, 0].set_ylabel('Stability Probability')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].hist(preds_df['final_score'], bins=30, color='purple', edgecolor='k', alpha=0.7)
    axs[1, 1].set_title('(d) Distribution of Final Multi-Objective Scores')
    axs[1, 1].set_xlabel('Final Score')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'composite_analysis_1_' + timestamp + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    print('Composite plot saved to ' + plot_filepath)
    print('\n--- Sensitivity Analysis: tau vs Stability and Mechanical Properties ---')
    metrics = ['stability_prob', 'K_VRH_pred', 'G_VRH_pred', 'final_score']
    print('\nCorrelations with tau:')
    for m in metrics:
        p_corr, _ = pearsonr(preds_df['tau'], preds_df[m])
        s_corr, _ = spearmanr(preds_df['tau'], preds_df[m])
        print('  ' + m + ': Pearson r = ' + str(round(p_corr, 4)) + ', Spearman rho = ' + str(round(s_corr, 4)))
    bins = [0, 0.8, 0.9, 1.0, np.inf]
    labels = ['< 0.8', '0.8 - 0.9', '0.9 - 1.0', '> 1.0']
    preds_df['tau_bin'] = pd.cut(preds_df['tau'], bins=bins, labels=labels)
    binned_stats = preds_df.groupby('tau_bin', observed=False).agg(count=('tau', 'count'), mean_stability_prob=('stability_prob', 'mean'), mean_K_VRH=('K_VRH_pred', 'mean'), mean_G_VRH=('G_VRH_pred', 'mean'), mean_final_score=('final_score', 'mean'), viable_fraction=('mechanical_viability', 'mean')).reset_index()
    print('\nBinned Statistics by tau:')
    print(binned_stats.to_string(index=False))