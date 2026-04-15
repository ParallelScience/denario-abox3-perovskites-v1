# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    filepath = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(filepath)
    base_features = ['density', 'volume', 'spacegroup_number', 'A_Z', 'B_Z', 'A_en', 'B_en', 'A_radius', 'B_radius', 'tau', 'mu', 'en_diff']
    crystal_system_cols = [col for col in df.columns if col.startswith('crystal_system_')]
    features = base_features + crystal_system_cols
    X = df[features]
    y_metal = df['is_metal'].astype(int)
    print('Class balance for is_metal:')
    class_counts = y_metal.value_counts()
    for cls_val, count in class_counts.items():
        label = 'Metal' if cls_val == 1 else 'Non-Metal'
        print('Class ' + str(cls_val) + ' (' + label + '): ' + str(count) + ' samples')
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)
    cv_results_clf = cross_validate(clf, X, y_metal, cv=5, scoring='roc_auc', n_jobs=4)
    roc_auc_mean = np.mean(cv_results_clf['test_score'])
    roc_auc_std = np.std(cv_results_clf['test_score'])
    print('Cross-validated ROC-AUC for is_metal: ' + str(round(roc_auc_mean, 4)) + ' +/- ' + str(round(roc_auc_std, 4)))
    y_metal_pred_cv = cross_val_predict(clf, X, y_metal, cv=5, n_jobs=4)
    cm = confusion_matrix(y_metal, y_metal_pred_cv)
    non_metal_mask = df['band_gap'] > 0
    X_non_metal = df.loc[non_metal_mask, features]
    y_bg = df.loc[non_metal_mask, 'band_gap']
    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
    y_bg_pred_cv = cross_val_predict(reg, X_non_metal, y_bg, cv=5, n_jobs=4)
    rmse = np.sqrt(mean_squared_error(y_bg, y_bg_pred_cv))
    mae = mean_absolute_error(y_bg, y_bg_pred_cv)
    print('Non-metallic subset Band Gap RMSE: ' + str(round(rmse, 4)) + ' eV')
    print('Non-metallic subset Band Gap MAE: ' + str(round(mae, 4)) + ' eV')
    clf.fit(X, y_metal)
    reg.fit(X_non_metal, y_bg)
    df['predicted_is_metal'] = clf.predict(X)
    df['predicted_band_gap'] = 0.0
    pred_non_metal_mask = df['predicted_is_metal'] == 0
    df.loc[pred_non_metal_mask, 'predicted_band_gap'] = reg.predict(df.loc[pred_non_metal_mask, features])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Metal', 'Metal'])
    disp.plot(ax=ax1, cmap='Blues', colorbar=False)
    ax1.set_title('Confusion Matrix: is_metal Classifier (CV)')
    ax2.scatter(y_bg, y_bg_pred_cv, alpha=0.6, edgecolors='k', color='orange')
    max_val = max(y_bg.max(), y_bg_pred_cv.max()) + 0.5
    ax2.plot([0, max_val], [0, max_val], 'r--', lw=2)
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    ax2.set_xlabel('Actual Band Gap (eV)')
    ax2.set_ylabel('Predicted Band Gap (eV)')
    ax2.set_title('RF Regressor: Band Gap (Non-Metals CV)')
    textstr = 'RMSE: ' + str(round(rmse, 3)) + ' eV\nMAE: ' + str(round(mae, 3)) + ' eV'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, verticalalignment='top', bbox=props)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'electronic_hurdle_diagnostic_1_' + timestamp + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print('Diagnostic plot saved to ' + plot_filepath)
    df.to_csv(filepath, index=False)
    print('Dataset with predicted band gap values saved to ' + filepath)

if __name__ == '__main__':
    main()