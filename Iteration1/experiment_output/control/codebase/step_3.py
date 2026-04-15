# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import time

def train_optoelectronic_models():
    data_path = 'data/cleaned_perovskite_data.csv'
    df = pd.read_csv(data_path, index_col='material_id')
    features = ['A_Z', 'B_Z', 'A_radius', 'B_radius', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'A_group', 'B_group', 'en_diff', 'tau', 'mu', 'B_valence', 'log_volume', 'abs_tau_diff', 'radius_diff', 'ie_ratio']
    X = df[features]
    y_class = df['is_metal'].astype(int)
    clf = GradientBoostingClassifier(random_state=42)
    cv_class = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_class = cross_val_predict(clf, X, y_class, cv=cv_class)
    y_proba_class = cross_val_predict(clf, X, y_class, cv=cv_class, method='predict_proba')[:, 1]
    acc = accuracy_score(y_class, y_pred_class)
    f1 = f1_score(y_class, y_pred_class)
    roc_auc = roc_auc_score(y_class, y_proba_class)
    print('========================================')
    print('--- CLASSIFICATION METRICS (is_metal) ---')
    print('========================================')
    print('Accuracy: ' + str(round(acc, 4)))
    print('F1 Score: ' + str(round(f1, 4)))
    print('ROC-AUC:  ' + str(round(roc_auc, 4)))
    non_metal_mask = df['is_metal'] == False
    X_reg = df.loc[non_metal_mask, features]
    y_reg = df.loc[non_metal_mask, 'band_gap']
    reg = GradientBoostingRegressor(random_state=42)
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_reg = cross_val_predict(reg, X_reg, y_reg, cv=cv_reg)
    mae = mean_absolute_error(y_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_reg, y_pred_reg))
    r2 = r2_score(y_reg, y_pred_reg)
    print('\n========================================')
    print('--- REGRESSION METRICS (band_gap) ---')
    print('========================================')
    print('MAE:  ' + str(round(mae, 4)) + ' eV')
    print('RMSE: ' + str(round(rmse, 4)) + ' eV')
    print('R2:   ' + str(round(r2, 4)))
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cm = confusion_matrix(y_class, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Metal', 'Metal'])
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Confusion Matrix: is_metal')
    axes[1].scatter(y_reg, y_pred_reg, alpha=0.6, edgecolors='k')
    axes[1].plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Band Gap (eV)')
    axes[1].set_ylabel('Predicted Band Gap (eV)')
    axes[1].set_title('Predicted vs Actual Band Gap (Non-Metals)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join('data', 'optoelectronic_hurdle_1_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print('\nPlot saved to ' + plot_filename)
    clf.fit(X, y_class)
    reg.fit(X_reg, y_reg)
    clf_filename = os.path.join('data', 'is_metal_classifier.joblib')
    reg_filename = os.path.join('data', 'band_gap_regressor.joblib')
    joblib.dump(clf, clf_filename)
    joblib.dump(reg, reg_filename)
    print('Trained classifier saved to ' + clf_filename)
    print('Trained regressor saved to ' + reg_filename)

if __name__ == '__main__':
    train_optoelectronic_models()