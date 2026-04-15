# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time

def train_and_evaluate(X, y, skf, gbc):
    oof_preds = np.zeros(len(X))
    feature_importances = np.zeros(X.shape[1])
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        gbc.fit(X_train, y_train)
        oof_preds[test_idx] = gbc.predict_proba(X_test)[:, 1]
        feature_importances += gbc.feature_importances_ / skf.n_splits
    return oof_preds, feature_importances

def main():
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'cleaned_dataset.csv')
    print('Loading dataset from: ' + data_path)
    df = pd.read_csv(data_path)
    if 'material_id' in df.columns:
        df = df.set_index('material_id')
    y = (df['energy_above_hull'] == 0).astype(int)
    print('Target is_stable defined. Positive class (stable) count: ' + str(y.sum()) + ', Negative class count: ' + str(len(y) - y.sum()))
    exclude_cols = ['energy_above_hull', 'formation_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'is_stable', 'band_gap', 'is_gap_direct', 'is_metal', 'efermi', 'is_magnetic', 'total_magnetization', 'total_magnetization_per_fu', 'num_magnetic_sites', 'K_VRH', 'K_voigt', 'K_reuss', 'G_VRH', 'G_voigt', 'G_reuss', 'elastic_anisotropy', 'poisson_ratio', 'pugh_ratio', 'energy_per_atom', 'formula', 'chemsys', 'spacegroup_symbol']
    mag_cols = [col for col in df.columns if col.startswith('magnetic_ordering_')]
    exclude_cols.extend(mag_cols)
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    print('Initial feature matrix shape: ' + str(X.shape))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    geom_proxies = ['volume', 'density', 'density_atomic']
    geom_cols_present = [col for col in geom_proxies if col in X.columns]
    print('\n--- Feature Importance Sensitivity Analysis ---')
    oof_preds_all, _ = train_and_evaluate(X, y, skf, gbc)
    roc_auc_all = roc_auc_score(y, oof_preds_all)
    print('ROC AUC with geometric features (' + ', '.join(geom_cols_present) + '): ' + str(round(roc_auc_all, 4)))
    X_no_geom = X.drop(columns=geom_cols_present)
    oof_preds_no_geom, importances_no_geom = train_and_evaluate(X_no_geom, y, skf, gbc)
    roc_auc_no_geom = roc_auc_score(y, oof_preds_no_geom)
    print('ROC AUC without geometric features: ' + str(round(roc_auc_no_geom, 4)))
    if roc_auc_all - roc_auc_no_geom < 0.05:
        print('Performance is stable without geometric features (drop < 0.05). Prioritizing compositional/bonding features.')
        final_X = X_no_geom
        final_oof_preds = oof_preds_no_geom
        final_importances = importances_no_geom
        final_roc_auc = roc_auc_no_geom
    else:
        print('Significant performance drop without geometric features. Retaining them.')
        final_X = X
        final_oof_preds = oof_preds_all
        _, final_importances = train_and_evaluate(final_X, y, skf, gbc)
        final_roc_auc = roc_auc_all
    y_pred_bin = (final_oof_preds >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred_bin)
    prec = precision_score(y, y_pred_bin, zero_division=0)
    rec = recall_score(y, y_pred_bin, zero_division=0)
    f1 = f1_score(y, y_pred_bin, zero_division=0)
    print('\n--- Final Thermodynamic Stability Classification Metrics ---')
    print('ROC AUC: ' + str(round(final_roc_auc, 4)))
    print('Accuracy: ' + str(round(acc, 4)))
    print('Precision: ' + str(round(prec, 4)))
    print('Recall: ' + str(round(rec, 4)))
    print('F1 Score: ' + str(round(f1, 4)))
    plt.rcParams['text.usetex'] = False
    fpr, tpr, _ = roc_curve(y, final_oof_preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = ' + str(round(final_roc_auc, 2)) + ')')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    fi_df = pd.DataFrame({'Feature': final_X.columns, 'Importance': final_importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)
    axes[1].barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color='skyblue')
    axes[1].set_xlabel('Mean Decrease in Impurity')
    axes[1].set_title('Top 20 Feature Importances')
    axes[1].grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = os.path.join(data_dir, 'stability_roc_fi_2_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('\nPlot saved to ' + plot_filename)
    preds_df = pd.DataFrame({'material_id': df.index, 'is_stable_actual': y, 'is_stable_prob': final_oof_preds, 'is_stable_pred': y_pred_bin})
    preds_path = os.path.join(data_dir, 'stability_predictions.csv')
    preds_df.to_csv(preds_path, index=False)
    print('Predictions saved to ' + preds_path)

if __name__ == '__main__':
    main()