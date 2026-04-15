# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    filepath = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(filepath)
    base_features = ['density', 'A_Z', 'B_Z', 'A_en', 'B_en', 'A_radius', 'B_radius', 'tau', 'mu', 'VEC', 'en_diff']
    crystal_system_cols = [col for col in df.columns if col.startswith('crystal_system_')]
    features = base_features + crystal_system_cols
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    train_mask = ~df['K_VRH'].isna()
    uncharacterized_mask = df['K_VRH'].isna()
    df_train = df[train_mask]
    df_uncharacterized = df[uncharacterized_mask].copy()
    df_uncharacterized['stability_rank'] = df_uncharacterized['predicted_is_stable_prob'].rank(ascending=False, method='min').astype(int)
    df_nm = df_uncharacterized[df_uncharacterized['predicted_is_metal'] == 0].copy()
    df_nm = df_nm.sort_values('predicted_is_stable_prob', ascending=False)
    top_n = 100
    candidates = df_nm.head(top_n)
    chemsys_counts = candidates['chemsys'].value_counts()
    final_candidates = candidates.groupby('chemsys').head(1).reset_index(drop=True)
    final_candidates = final_candidates.sort_values('predicted_is_stable_prob', ascending=False)
    print('Total uncharacterized materials: ' + str(len(df_uncharacterized)))
    print('Non-metallic uncharacterized materials: ' + str(len(df_nm)))
    print('Selecting top ' + str(top_n) + ' non-metallic candidates based on stability probability.')
    print('Number of unique chemical systems in top ' + str(top_n) + ': ' + str(len(chemsys_counts)))
    print('Number of final diverse candidates: ' + str(len(final_candidates)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(df_uncharacterized['PC1'], df_uncharacterized['PC2'], alpha=0.4, c='lightgray', edgecolors='k', label='Uncharacterized (' + str(len(df_uncharacterized)) + ')', s=30)
    ax1.scatter(df_train['PC1'], df_train['PC2'], alpha=0.8, c='blue', edgecolors='k', label='Training Set (' + str(len(df_train)) + ')', s=40)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_features_idx = np.argsort(loading_magnitudes)[-6:]
    max_scatter = max(np.max(np.abs(X_pca[:, 0])), np.max(np.abs(X_pca[:, 1])))
    max_loading = np.max(np.abs(loadings[top_features_idx, :2]))
    scaling_factor = (max_scatter * 0.5) / max_loading if max_loading > 0 else 1
    for i in top_features_idx:
        ax1.arrow(0, 0, loadings[i, 0]*scaling_factor, loadings[i, 1]*scaling_factor, color='red', alpha=0.8, head_width=max_scatter*0.02, head_length=max_scatter*0.03, linewidth=1.5)
        feat_name = features[i].replace('crystal_system_', '')
        ax1.text(loadings[i, 0]*scaling_factor*1.15, loadings[i, 1]*scaling_factor*1.15, feat_name, color='darkred', fontsize=9, ha='center', va='center')
    ax1.set_xlabel('PC1 (' + str(round(pca.explained_variance_ratio_[0]*100, 1)) + '%)')
    ax1.set_ylabel('PC2 (' + str(round(pca.explained_variance_ratio_[1]*100, 1)) + '%)')
    ax1.set_title('PCA Biplot: Applicability Domain')
    ax1.legend()
    plot_counts = chemsys_counts.head(20)
    ax2.bar(plot_counts.index, plot_counts.values, color='coral', edgecolor='k')
    ax2.set_xlabel('Chemical System (chemsys)')
    ax2.set_ylabel('Number of Candidates (in Top ' + str(top_n) + ')')
    ax2.set_title('Top ' + str(top_n) + ' Candidates per Chemsys Cluster')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'applicability_diversity_1_' + timestamp + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print('Applicability and diversity plot saved to ' + plot_filepath)
    print('\nFinal High-Performance Candidates (Diverse, Non-Metallic, from Top ' + str(top_n) + '):')
    print('-' * 105)
    header = 'Formula'.ljust(12) + ' | ' + 'Material ID'.ljust(15) + ' | ' + 'K_VRH (GPa)'.ljust(12) + ' | ' + 'G_VRH (GPa)'.ljust(12) + ' | ' + 'Stab Rank'.ljust(10) + ' | ' + 'Band Gap (eV)'.ljust(15)
    print(header)
    print('-' * 105)
    for _, row in final_candidates.iterrows():
        form = str(row['formula']).ljust(12)
        mat_id = str(row['material_id']).ljust(15)
        k_vrh = str(round(row['predicted_K_VRH'], 2)).ljust(12)
        g_vrh = str(round(row['predicted_G_VRH'], 2)).ljust(12)
        rank = str(int(row['stability_rank'])).ljust(10)
        bg = str(round(row['predicted_band_gap'], 4)).ljust(15)
        print(form + ' | ' + mat_id + ' | ' + k_vrh + ' | ' + g_vrh + ' | ' + rank + ' | ' + bg)
    print('-' * 105 + '\n')

if __name__ == '__main__':
    main()