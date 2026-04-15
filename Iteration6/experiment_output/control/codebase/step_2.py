# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import joblib

def train_stability_model():
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_perovskite_data.csv'))
    features = ['tau_strain', 'mu_strain', 'tau', 'mu', 'en_diff', 'glazer_tilt']
    X = df[features].copy()
    y = df['is_stable'].astype(int)
    groups = df['A_site']
    X = pd.get_dummies(X, columns=['glazer_tilt'], drop_first=False).astype(float)
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(X, y, groups)
    print('Number of folds (unique A_site groups): ' + str(n_folds))
    y_true_all = []
    y_prob_all = []
    y_pred_all = []
    fold = 1
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        n_stable = int(y_test.sum())
        n_total = len(y_test)
        group_name = str(groups.iloc[test_idx].iloc[0])
        pct_stable = round((n_stable / n_total) * 100, 1)
        print('Fold ' + str(fold) + ' (Group ' + group_name + '): ' + str(n_stable) + '/' + str(n_total) + ' stable (' + str(pct_stable) + '%)')
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob)
        y_pred_all.extend(y_pred)
        fold += 1
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    y_pred_all = np.array(y_pred_all)
    auc_pr = average_precision_score(y_true_all, y_prob_all)
    f1 = f1_score(y_true_all, y_pred_all)
    print('\nOverall LOCO CV Metrics:')
    print('AUC-PR: ' + str(round(auc_pr, 4)))
    print('F1 Score: ' + str(round(f1, 4)))
    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='GBC (AUC-PR = ' + str(round(auc_pr, 3)) + ')', color='blue')
    plt.axhline(y=0.131, color='red', linestyle='--', label='No-skill baseline (0.131)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (LOCO CV)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'pr_curve_stability_1_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print('Precision-Recall curve saved to ' + plot_filename)
    final_clf = GradientBoostingClassifier(random_state=42)
    final_clf.fit(X, y)
    model_path = os.path.join(data_dir, 'stability_gbc_model.joblib')
    joblib.dump(final_clf, model_path)
    print('Final model saved to ' + model_path)
    feature_names_path = os.path.join(data_dir, 'stability_gbc_features.joblib')
    joblib.dump(list(X.columns), feature_names_path)
    metrics = {'AUC-PR': auc_pr, 'F1': f1}
    metrics_path = os.path.join(data_dir, 'stability_metrics.joblib')
    joblib.dump(metrics, metrics_path)
    print('Metrics saved to ' + metrics_path)

if __name__ == '__main__':
    train_stability_model()