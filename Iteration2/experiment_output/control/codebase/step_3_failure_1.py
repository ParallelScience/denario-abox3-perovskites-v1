# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

def main():
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'cleaned_dataset.csv')
    print('Loading dataset from: ' + data_path)
    df = pd.read_csv(data_path)
    if 'material_id' in df.columns:
        df = df.set_index('material_id')
    df_elastic = df.dropna(subset=['K_VRH', 'G_VRH']).copy()
    print('Elastic subset size: ' + str(len(df_elastic)))
    k_p1 = df_elastic['K_VRH'].quantile(0.01)
    k_p99 = df_elastic['K_VRH'].quantile(0.99)
    g_p1 = df_elastic['G_VRH'].quantile(0.01)
    g_p99 = df_elastic['G_VRH'].quantile(0.99)
    print('K_VRH 1st-99th percentile range: ' + str(round(k_p1, 2)) + ' to ' + str(round(k_p99, 2)) + ' GPa')
    print('G_VRH 1st-99th percentile range: ' + str(round(g_p1, 2)) + ' to ' + str(round(g_p99, 2)) + ' GPa')
    df_elastic['is_viable'] = ((df_elastic['K_VRH'] >= k_p1) & (df_elastic['K_VRH'] <= k_p99) & (df_elastic['G_VRH'] >= g_p1) & (df_elastic['G_VRH'] <= g_p99)).astype(int)
    n_viable = df_elastic['is_viable'].sum()
    n_unstable = len(df_elastic) - n_viable
    print('Class distribution - Viable (1): ' + str(n_viable) + ', Unstable/Pathological (0): ' + str(n_unstable))
    exclude_cols = ['energy_above_hull', 'formation_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'is_stable', 'band_gap', 'is_gap_direct', 'is_metal', 'efermi', 'is_magnetic', 'total_magnetization', 'total_magnetization_per_fu', 'num_magnetic_sites', 'K_VRH', 'K_voigt', 'K_reuss', 'G_VRH', 'G_voigt', 'G_reuss', 'elastic_anisotropy', 'poisson_ratio', 'pugh_ratio', 'energy_per_atom', 'formula', 'chemsys', 'spacegroup_symbol']
    mag_cols = [col for col in df.columns if col.startswith('magnetic_ordering_')]
    exclude_cols.extend(mag_cols)
    X_all = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    X_all = X_all.select_dtypes(include=[np.number])
    X_all = X_all.fillna(X_all.median())
    print('Number of features used for training: ' + str(X_all.shape[1]))
    X_elastic = X_all.loc[df_elastic.index]
    y_elastic = df_elastic['is_viable']
    n_minority = min(n_viable, n_unstable)
    n_splits = min(5, max(2, n_minority))
    print('Using StratifiedKFold with n_splits=' + str(n_splits))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    oof_preds = np.zeros(len(X_elastic))
    for train_idx, test_idx in skf.split(X_elastic, y_elastic):
        X_train, X_test = X_elastic.iloc[train_idx], X_elastic.iloc[test_idx]
        y_train, y_test = y_elastic.iloc[train_idx], y_elastic.iloc[test_idx]
        weights = compute_sample_weight(class_weight='balanced', y=y_train)
        gbc.fit(X_train, y_train, sample_weight=weights)
        oof_preds[test_idx] = gbc.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_elastic, oof_preds)
    pr_auc = average_precision_score(y_elastic, oof_preds)
    y_pred_bin = (oof_preds >= 0.5).astype(int)
    cm = confusion_matrix(y_elastic, y_pred_bin)
    print('\n--- Mechanical Viability Classification Metrics (OOF) ---')
    print('ROC AUC: ' + str(round(roc_auc, 4)))
    print('PR AUC: ' + str(round(pr_auc, 4)))
    print('Confusion Matrix:\n' + str(cm))
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fpr, tpr, _ = roc_curve(y_elastic, oof_preds)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = ' + str(round(roc_auc, 2)) + ')')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    precision, recall, _ = precision_recall_curve(y_elastic, oof_preds)
    axes[1].plot(recall, precision, color='blue', lw=2, label='PR curve (AUC = ' + str(round(pr_auc, 2)) + ')')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    axes[2].matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(x=j, y=i, s=str(cm[i, j]), va='center', ha='center', size='xx-large')
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_ylabel('True Label')
    axes[2].set_title('Confusion Matrix', pad=20)
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(['Unstable', 'Viable'])
    axes[2].set_yticklabels(['Unstable', 'Viable'])
    axes[2].xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = os.path.join(data_dir, 'mechanical_viability_eval_3_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('\nPlot saved to ' + plot_filename)
    weights_all = compute_sample_weight(class_weight='balanced', y=y_elastic)
    gbc.fit(X_elastic, y_elastic, sample_weight=weights_all)
    all_preds = gbc.predict_proba(X_all)[:, 1]
    preds_df = pd.DataFrame({'material_id': df.index, 'is_viable_prob': all_preds, 'is_viable_pred': (all_preds >= 0.5).astype(int)})
    preds_df['is_viable_actual'] = np.nan
    preds_df.loc[df_elastic.index, 'is_viable_actual'] = y_elastic
    preds_path = os.path.join(data_dir, 'mechanical_viability_predictions.csv')
    preds_df.to_csv(preds_path, index=False)
    print('Predictions saved to ' + preds_path)

if __name__ == '__main__':
    main()