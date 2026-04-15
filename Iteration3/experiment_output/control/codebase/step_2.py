# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
plt.rcParams['text.usetex'] = False
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    features = ['nsites', 'volume', 'density', 'density_atomic', 'spacegroup_number', 'A_Z', 'A_radius', 'A_en', 'A_ie1', 'A_group', 'B_Z', 'B_radius', 'B_en', 'B_ie1', 'B_group', 'B_valence', 'tau', 'mu', 'en_diff', 'VEC', 'volume_residual']
    target = 'is_stable_soft'
    X = df[features]
    y = df[target].astype(int)
    return X, y
def train_and_evaluate_model(X, y):
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    gbc = GradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_probs = np.zeros(len(y))
    y_preds = np.zeros(len(y))
    feature_importances = np.zeros(X.shape[1])
    for train_idx, test_idx in cv.split(X_imputed, y):
        X_train = X_imputed.iloc[train_idx]
        X_test = X_imputed.iloc[test_idx]
        y_train = y.iloc[train_idx]
        sw_train = sample_weights[train_idx]
        gbc.fit(X_train, y_train, sample_weight=sw_train)
        y_probs[test_idx] = gbc.predict_proba(X_test)[:, 1]
        y_preds[test_idx] = gbc.predict(X_test)
        feature_importances += gbc.feature_importances_ / cv.n_splits
    return y_probs, y_preds, feature_importances
def plot_roc_curve(y_true, y_probs, roc_auc, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='GBC (AUC = ' + str(round(roc_auc, 4)) + ')')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Thermodynamic Stability')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    filepath = os.path.join(output_dir, 'roc_curve_1_' + timestamp + '.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    print('ROC curve saved to ' + filepath)
def plot_pr_curve(y_true, y_probs, ap_score, output_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='GBC (AP = ' + str(round(ap_score, 4)) + ')')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Thermodynamic Stability')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    filepath = os.path.join(output_dir, 'pr_curve_2_' + timestamp + '.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    print('Precision-Recall curve saved to ' + filepath)
def main():
    data_dir = 'data/'
    input_filepath = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    X, y = load_and_prepare_data(input_filepath)
    print('Class distribution (normalized):')
    print(y.value_counts(normalize=True).to_string())
    print('\nClass distribution (counts):')
    print(y.value_counts().to_string())
    y_probs, y_preds, feature_importances = train_and_evaluate_model(X, y)
    cm = confusion_matrix(y, y_preds)
    roc_auc = roc_auc_score(y, y_probs)
    ap_score = average_precision_score(y, y_probs)
    print('\n--- Cross-Validation Results ---')
    print('AUC-ROC: ' + str(round(roc_auc, 4)))
    print('Average Precision (PR-AUC): ' + str(round(ap_score, 4)))
    print('\nConfusion Matrix:')
    print(str(cm))
    print('\nClassification Report:')
    print(classification_report(y, y_preds))
    fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
    print('\nFeature Importances:')
    print(fi_df.to_string(index=False))
    metrics_df = pd.DataFrame({'Metric': ['AUC-ROC', 'Average Precision'], 'Value': [roc_auc, ap_score]})
    metrics_df.to_csv(os.path.join(data_dir, 'gbc_metrics.csv'), index=False)
    fi_df.to_csv(os.path.join(data_dir, 'gbc_feature_importance.csv'), index=False)
    plot_roc_curve(y, y_probs, roc_auc, data_dir)
    plot_pr_curve(y, y_probs, ap_score, data_dir)
if __name__ == '__main__':
    main()