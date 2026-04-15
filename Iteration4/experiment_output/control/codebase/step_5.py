# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error

def main():
    data_dir = "data/"
    input_file = os.path.join(data_dir, "classification_results.csv")
    df = pd.read_csv(input_file)
    features = ['tau', 'mu', 'volume', 'volume_residual', 'en_diff', 'A_Z', 'B_Z', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'B_valence', 'VEC', 'tilt_proxy']
    if 'is_metal' not in df.columns:
        df['is_metal'] = df['band_gap'] == 0
    else:
        df['is_metal'] = df['is_metal'].astype(bool)
    print("Electronic Property Hurdle Modeling")
    print("=============================================")
    class_counts = df['is_metal'].value_counts()
    print("Class balance for metallicity classifier:")
    print("Metals (is_metal=True): " + str(class_counts.get(True, 0)))
    print("Non-metals (is_metal=False): " + str(class_counts.get(False, 0)))
    X_class = df[features]
    y_class = df['is_metal']
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    cv_class = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_class_pred = cross_val_predict(clf, X_class, y_class, cv=cv_class)
    clf.fit(X_class, y_class)
    df['predicted_is_metal'] = clf.predict(X_class)
    df['metallicity_probability'] = clf.predict_proba(X_class)[:, 1]
    non_metals = df[~df['is_metal']].copy()
    X_reg = non_metals[features]
    y_reg = non_metals['band_gap']
    reg = RandomForestRegressor(random_state=42)
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    y_reg_pred = cross_val_predict(reg, X_reg, y_reg, cv=cv_reg)
    rmse = np.sqrt(mean_squared_error(y_reg, y_reg_pred))
    mae = mean_absolute_error(y_reg, y_reg_pred)
    print("\nBand Gap Regressor Performance (Non-metals, 5-fold CV):")
    print("RMSE: " + str(round(rmse, 4)) + " eV")
    print("MAE: " + str(round(mae, 4)) + " eV")
    reg.fit(X_reg, y_reg)
    df['predicted_band_gap_continuous'] = reg.predict(df[features])
    df['predicted_band_gap'] = np.where(df['predicted_is_metal'], 0.0, df['predicted_band_gap_continuous'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cm = confusion_matrix(y_class, y_class_pred, labels=[False, True])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-metal', 'Metal'])
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Metallicity Classifier Confusion Matrix')
    axes[1].scatter(y_reg, y_reg_pred, alpha=0.6, edgecolors='k', color='teal')
    min_val = min(y_reg.min(), y_reg_pred.min())
    max_val = max(y_reg.max(), y_reg_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Reference')
    axes[1].set_xlabel('Actual Band Gap (eV)')
    axes[1].set_ylabel('Predicted Band Gap (eV)')
    axes[1].set_title('Non-Metallic Band Gap Regression')
    textstr = "RMSE = " + str(round(rmse, 3)) + " eV\nMAE = " + str(round(mae, 3)) + " eV"
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].legend(loc='lower right')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(data_dir, 'electronic_hurdle_5_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    print("\nCombined figure saved to " + plot_filename)
    clf_filename = os.path.join(data_dir, 'metallicity_model.joblib')
    reg_filename = os.path.join(data_dir, 'band_gap_model.joblib')
    joblib.dump(clf, clf_filename)
    joblib.dump(reg, reg_filename)
    print("Models saved to " + clf_filename + " and " + reg_filename)
    results_filename = os.path.join(data_dir, 'classification_results.csv')
    df.to_csv(results_filename, index=False)
    print("Updated classification results saved to " + results_filename)

if __name__ == '__main__':
    main()