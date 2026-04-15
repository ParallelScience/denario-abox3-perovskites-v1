# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score

def train_hurdle_model():
    data_dir = "data/"
    input_path = os.path.join(data_dir, "cleaned_perovskite_data.csv")
    print("Loading dataset from " + input_path + "...")
    df = pd.read_csv(input_path)
    base_features = ['density', 'tau', 'mu', 'tau_strain', 'mu_strain', 'en_diff', 'A_radius', 'B_radius', 'formation_energy_per_atom']
    crystal_dummies = pd.get_dummies(df['crystal_system'], prefix='crystal', drop_first=False).astype(float)
    glazer_dummies = pd.get_dummies(df['glazer_tilt'], prefix='tilt', drop_first=False).astype(float)
    X = pd.concat([df[base_features], crystal_dummies, glazer_dummies], axis=1)
    feature_names = X.columns.tolist()
    y_is_metal = df['is_metal'].astype(int)
    y_gap = df['band_gap']
    X_train, X_test, y_metal_train, y_metal_test, y_gap_train, y_gap_test = train_test_split(X, y_is_metal, y_gap, test_size=0.2, random_state=42)
    print("Total samples: " + str(len(df)))
    print("Training samples: " + str(len(X_train)) + ", Testing samples: " + str(len(X_test)))
    print("Number of features: " + str(X.shape[1]))
    print("\nTraining GradientBoostingClassifier for is_metal...")
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_metal_train)
    metal_preds = clf.predict(X_test)
    metal_probs = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_metal_test, metal_preds)
    auc = roc_auc_score(y_metal_test, metal_probs)
    print("Classifier Accuracy: " + str(round(acc, 4)))
    print("Classifier ROC-AUC: " + str(round(auc, 4)))
    print("\nTraining GradientBoostingRegressor on log1p-transformed non-zero band gaps...")
    non_metal_train_mask = (y_gap_train > 0)
    X_train_reg = X_train[non_metal_train_mask]
    y_gap_train_reg = y_gap_train[non_metal_train_mask]
    y_gap_train_log = np.log1p(y_gap_train_reg)
    reg = GradientBoostingRegressor(random_state=42)
    reg.fit(X_train_reg, y_gap_train_log)
    non_metal_test_mask = (y_gap_test > 0)
    X_test_reg = X_test[non_metal_test_mask]
    y_gap_test_reg = y_gap_test[non_metal_test_mask]
    y_gap_test_log = np.log1p(y_gap_test_reg)
    preds_log = reg.predict(X_test_reg)
    preds_ev = np.expm1(preds_log)
    rmse_log = np.sqrt(mean_squared_error(y_gap_test_log, preds_log))
    mae_ev = mean_absolute_error(y_gap_test_reg, preds_ev)
    print("Regressor Evaluation (on actual non-metals in test set):")
    print("RMSE (log scale): " + str(round(rmse_log, 4)))
    print("MAE (original eV scale): " + str(round(mae_ev, 4)) + " eV")
    print("\nEvaluating full Hurdle Model...")
    pred_is_metal = clf.predict(X_test)
    pred_gap_log = reg.predict(X_test)
    pred_gap_ev = np.expm1(pred_gap_log)
    final_pred_gap = np.where(pred_is_metal == 1, 0.0, pred_gap_ev)
    overall_mae = mean_absolute_error(y_gap_test, final_pred_gap)
    print("Overall Hurdle Model MAE: " + str(round(overall_mae, 4)) + " eV")
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 8))
    plt.scatter(y_gap_test, final_pred_gap, alpha=0.6, edgecolors='k', s=40, label='Predictions')
    max_val = max(y_gap_test.max(), final_pred_gap.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Agreement')
    plt.xlabel('Actual Band Gap (eV)')
    plt.ylabel('Predicted Band Gap (eV)')
    plt.title('Hurdle Model: Predicted vs Actual Band Gap')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'parity_plot_bandgap_1_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print("Parity plot saved to " + plot_filename)
    print("\nRetraining models on full dataset for final saving...")
    clf_final = GradientBoostingClassifier(random_state=42)
    clf_final.fit(X, y_is_metal)
    non_metal_mask = (y_gap > 0)
    X_reg = X[non_metal_mask]
    y_gap_reg = y_gap[non_metal_mask]
    y_gap_log = np.log1p(y_gap_reg)
    reg_final = GradientBoostingRegressor(random_state=42)
    reg_final.fit(X_reg, y_gap_log)
    joblib.dump(clf_final, os.path.join(data_dir, 'hurdle_classifier_is_metal.joblib'))
    joblib.dump(reg_final, os.path.join(data_dir, 'hurdle_regressor_bandgap.joblib'))
    joblib.dump(feature_names, os.path.join(data_dir, 'hurdle_features.joblib'))
    metrics = {'classifier_accuracy': acc, 'classifier_roc_auc': auc, 'regressor_rmse_log': rmse_log, 'regressor_mae_ev': mae_ev, 'overall_mae_ev': overall_mae}
    joblib.dump(metrics, os.path.join(data_dir, 'hurdle_metrics.joblib'))
    print("Models and metrics saved to " + data_dir)

if __name__ == '__main__':
    train_hurdle_model()