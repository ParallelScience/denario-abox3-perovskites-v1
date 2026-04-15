# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

def identify_pareto(scores):
    n = scores.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        diff = scores - scores[i]
        dominates_i = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
        if np.any(dominates_i):
            is_pareto[i] = False
    return is_pareto

def main():
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_dataset.csv'))
    if 'material_id' in df.columns:
        df = df.set_index('material_id')
    df_stab = pd.read_csv(os.path.join(data_dir, 'stability_predictions.csv'))
    if 'material_id' in df_stab.columns:
        df_stab = df_stab.set_index('material_id')
    df_mech = pd.read_csv(os.path.join(data_dir, 'mechanical_viability_predictions.csv'))
    if 'material_id' in df_mech.columns:
        df_mech = df_mech.set_index('material_id')
    df_elec = pd.read_csv(os.path.join(data_dir, 'electronic_ductility_predictions.csv'))
    if 'material_id' in df_elec.columns:
        df_elec = df_elec.set_index('material_id')
    exclude_cols = ['energy_above_hull', 'formation_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'is_stable', 'band_gap', 'is_gap_direct', 'is_metal', 'efermi', 'is_magnetic', 'total_magnetization', 'total_magnetization_per_fu', 'num_magnetic_sites', 'K_VRH', 'K_voigt', 'K_reuss', 'G_VRH', 'G_voigt', 'G_reuss', 'elastic_anisotropy', 'poisson_ratio', 'pugh_ratio', 'energy_per_atom', 'formula', 'chemsys', 'spacegroup_symbol']
    mag_cols = [col for col in df.columns if col.startswith('magnetic_ordering_')]
    exclude_cols.extend(mag_cols)
    X_all = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    X_all = X_all.select_dtypes(include=[np.number])
    X_all = X_all.fillna(X_all.median())
    df_elastic = df.dropna(subset=['K_VRH', 'G_VRH']).copy()
    X_elastic = X_all.loc[df_elastic.index]
    k_p1 = df_elastic['K_VRH'].quantile(0.01)
    k_p99 = df_elastic['K_VRH'].quantile(0.99)
    g_p1 = df_elastic['G_VRH'].quantile(0.01)
    g_p99 = df_elastic['G_VRH'].quantile(0.99)
    viable_mask = ((df_elastic['K_VRH'] >= k_p1) & (df_elastic['K_VRH'] <= k_p99) & (df_elastic['G_VRH'] >= g_p1) & (df_elastic['G_VRH'] <= g_p99))
    df_viable = df_elastic[viable_mask].copy()
    if 'pugh_ratio' in df_viable.columns and not df_viable['pugh_ratio'].isnull().all():
        pugh = df_viable['pugh_ratio']
    else:
        pugh = df_viable['G_VRH'] / df_viable['K_VRH']
    df_viable['is_ductile'] = (pugh < 0.571).astype(int)
    y_ductile = df_viable['is_ductile']
    X_ductile = X_all.loc[df_viable.index]
    gbc_ductile = GradientBoostingClassifier(n_estimators=100, random_state=42)
    weights = compute_sample_weight(class_weight='balanced', y=y_ductile)
    gbc_ductile.fit(X_ductile, y_ductile, sample_weight=weights)
    prob_duct = gbc_ductile.predict_proba(X_all)[:, 1]
    pareto_df = pd.DataFrame({'formula': df['formula'], 'chemsys': df['chemsys'], 'prob_stab': df_stab['is_stable_prob'], 'prob_mech': df_mech['is_viable_prob'], 'prob_duct': prob_duct, 'pred_is_metal': df_elec['pred_is_metal'], 'pred_band_gap': df_elec['pred_band_gap']}, index=df.index)
    scores = pareto_df[['prob_stab', 'prob_mech', 'prob_duct']].values
    pareto_mask = identify_pareto(scores)
    pareto_df['is_pareto'] = pareto_mask
    print('Total candidates: ' + str(len(pareto_df)))
    print('Number of Pareto-optimal candidates: ' + str(pareto_mask.sum()))
    mean_vec = np.mean(X_elastic.values, axis=0)
    cov_matrix = np.cov(X_elastic.values, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    m_dists = []
    for row in X_all.values:
        diff = row - mean_vec
        dist = np.sqrt(np.maximum(0, np.dot(np.dot(diff, inv_cov_matrix), diff.T)))
        m_dists.append(dist)
    pareto_df['mahalanobis_dist'] = m_dists
    train_m_dists = []
    for row in X_elastic.values:
        diff = row - mean_vec
        dist = np.sqrt(np.maximum(0, np.dot(np.dot(diff, inv_cov_matrix), diff.T)))
        train_m_dists.append(dist)
    threshold = np.percentile(train_m_dists, 97.5)
    pareto_df['is_high_novelty'] = pareto_df['mahalanobis_dist'] > threshold
    print('Mahalanobis distance 97.5th percentile threshold (training set): ' + str(round(threshold, 4)))
    print('Number of high-novelty candidates in full dataset: ' + str(pareto_df['is_high_novelty'].sum()))
    pareto_df['distance_to_ideal'] = np.sqrt((1 - pareto_df['prob_stab'])**2 + (1 - pareto_df['prob_mech'])**2 + (1 - pareto_df['prob_duct'])**2)
    pareto_df['is_diverse_pareto'] = False
    diverse_indices = pareto_df[pareto_df['is_pareto']].groupby('chemsys')['distance_to_ideal'].idxmin()
    pareto_df.loc[diverse_indices, 'is_diverse_pareto'] = True
    print('Number of diverse Pareto-optimal candidates: ' + str(len(diverse_indices)))
    pareto_df = pareto_df.sort_values('distance_to_ideal')
    print('\n--- Top 10 Diverse Pareto-Optimal Candidates ---')
    top_10_diverse = pareto_df[pareto_df['is_diverse_pareto']].head(10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(top_10_diverse[['formula', 'chemsys', 'prob_stab', 'prob_mech', 'prob_duct', 'distance_to_ideal', 'is_high_novelty']].to_string())
    plt.rcParams['text.usetex'] = False
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pareto_df['prob_stab'], pareto_df['prob_mech'], pareto_df['prob_duct'], c='lightgray', alpha=0.3, label='All Candidates', s=15)
    pareto_points = pareto_df[pareto_df['is_pareto']]
    unique_chemsys = pareto_points['chemsys'].unique()
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(unique_chemsys))]
    chemsys_color_map = dict(zip(unique_chemsys, colors))
    for chemsys in unique_chemsys:
        subset = pareto_points[pareto_points['chemsys'] == chemsys]
        ax1.scatter(subset['prob_stab'], subset['prob_mech'], subset['prob_duct'], c=[chemsys_color_map[chemsys]], label=chemsys, s=60, edgecolors='k', alpha=0.8)
    ax1.set_xlabel('Stability Prob')
    ax1.set_ylabel('Viability Prob')
    ax1.set_zlabel('Ductility Prob')
    ax1.set_title('3D Pareto Frontier')
    handles, labels = ax1.get_legend_handles_labels()
    if len(labels) > 16:
        ax1.legend(handles[:16], labels[:16], loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', title='Chemsys (Top 15)')
    else:
        ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', title='Chemsys')
    ax2 = fig.add_subplot(122)
    ax2.scatter(pareto_df['prob_stab'], pareto_df['prob_mech'], c='lightgray', alpha=0.3, label='All Candidates', s=15)
    sc2 = ax2.scatter(pareto_points['prob_stab'], pareto_points['prob_mech'], c=pareto_points['prob_duct'], cmap='viridis', s=60, edgecolors='k', alpha=0.9)
    ax2.set_xlabel('Stability Probability')
    ax2.set_ylabel('Mechanical Viability Probability')
    ax2.set_title('2D Projection of Pareto Frontier')
    cbar = plt.colorbar(sc2, ax=ax2)
    cbar.set_label('Ductility Probability')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = os.path.join(data_dir, 'pareto_frontier_5_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('\nPlot saved to ' + plot_filename)
    output_csv = os.path.join(data_dir, 'final_ranked_candidates.csv')
    pareto_df.to_csv(output_csv)
    print('Final ranked candidate table saved to ' + output_csv)

if __name__ == '__main__':
    main()