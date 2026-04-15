# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold
import xgboost as xgb

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    filepath = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(filepath)
    base_features = ['formation_energy_per_atom', 'tau', 'mu', 'en_diff', 'spacegroup_number', 'density', 'A_Z', 'B_Z', 'A_en', 'B_en']
    crystal_system_cols = [col for col in df.columns if col.startswith('crystal_system_')]
    features = base_features + crystal_system_cols
    X = df[features]
    y = df['is_stable'].astype(int)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', n_jobs=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['roc_auc', 'f1', 'average_precision']
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=4)
    roc_auc_mean = np.mean(cv_results['test_roc_auc'])
    roc_auc_std = np.std(cv_results['test_roc_auc'])
    f1_mean = np.mean(cv_results['test_f1'])
    f1_std = np.std(cv_results['test_f1'])
    ap_mean = np.mean(cv_results['test_average_precision'])
    ap_std = np.std(cv_results['test_average_precision'])
    print('Cross-Validation Results (5-fold):')
    print('ROC-AUC: ' + str(round(roc_auc_mean, 4)) + ' +/- ' + str(round(roc_auc_std, 4)))
    print('F1 Score: ' + str(round(f1_mean, 4)) + ' +/- ' + str(round(f1_std, 4)))
    print('Precision-Recall (Avg Precision): ' + str(round(ap_mean, 4)) + ' +/- ' + str(round(ap_std, 4)))
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_importances)), sorted_importances[::-1], align='center')
    plt.yticks(range(len(sorted_importances)), sorted_features[::-1])
    plt.xlabel('XGBoost Feature Importance (Relative)')
    plt.title('Feature Importances for Thermodynamic Stability Classification')
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'feature_importance_1_' + timestamp + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print('Feature importance plot saved to ' + plot_filepath)
    df['predicted_is_stable_prob'] = model.predict_proba(X)[:, 1]
    df.to_csv(filepath, index=False)
    print('Dataset with predicted stability probabilities saved to ' + filepath)

if __name__ == '__main__':
    main()