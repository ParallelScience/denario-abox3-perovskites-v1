# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import time

def train_ductility_model():
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(data_path, index_col='material_id')
    elastic_mask = df['is_elastic_characterized'] == True
    df_elastic = df[elastic_mask].copy()
    df_elastic['calculated_pugh_ratio'] = df_elastic['G_VRH'] / df_elastic['K_VRH']
    df_elastic['is_brittle'] = (df_elastic['calculated_pugh_ratio'] > 0.57).astype(int)
    print('========================================')
    print('--- DUCTILITY CLASS DISTRIBUTION ---')
    print('========================================')
    ductile_count = (df_elastic['is_brittle'] == 0).sum()
    brittle_count = (df_elastic['is_brittle'] == 1).sum()
    print('Total samples: ' + str(len(df_elastic)))
    print('Ductile (Pugh < 0.57): ' + str(ductile_count))
    print('Brittle (Pugh > 0.57): ' + str(brittle_count))
    features = ['A_Z', 'B_Z', 'A_radius', 'B_radius', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'A_group', 'B_group', 'en_diff', 'tau', 'mu', 'B_valence', 'log_volume', 'abs_tau_diff', 'radius_diff', 'ie_ratio']
    X_base = df_elastic[features].copy()
    clf_metal = joblib.load(os.path.join(data_dir, 'is_metal_classifier.joblib'))
    reg_bg = joblib.load(os.path.join(data_dir, 'band_gap_regressor.joblib'))
    pred_is_metal = clf_metal.predict(X_base)
    pred_band_gap = np.zeros(len(X_base))
    non_metal_mask = ~pred_is_metal.astype(bool)
    if non_metal_mask.sum() > 0:
        pred_band_gap[non_metal_mask] = reg_bg.predict(X_base[non_metal_mask])
    X_mech = X_base.copy()
    X_mech['pred_is_metal'] = pred_is_metal.astype(int)
    X_mech['pred_band_gap'] = pred_band_gap
    k_models = joblib.load(os.path.join(data_dir, 'k_vrh_quantiles_model.joblib'))
    g_models = joblib.load(os.path.join(data_dir, 'g_vrh_quantiles_model.joblib'))
    pred_k_vrh = k_models['q50'].predict(X_mech)
    pred_g_vrh = g_models['q50'].predict(X_mech)
    X_ductility = X_base.copy()
    X_ductility['pred_K_VRH'] = pred_k_vrh
    X_ductility['pred_G_VRH'] = pred_g_vrh
    y_ductility = df_elastic['is_brittle']
    clf_ductility = GradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf_ductility, X_ductility, y_ductility, cv=cv)
    acc = accuracy_score(y_ductility, y_pred)
    f1 = f1_score(y_ductility, y_pred)
    print('\n========================================')
    print('--- CLASSIFICATION METRICS (Ductility) ---')
    print('========================================')
    print('Accuracy: ' + str(round(acc, 4)))
    print('F1 Score (Brittle): ' + str(round(f1, 4)))
    plt.rcParams['text.usetex'] = False
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_ductility, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ductile', 'Brittle'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix: Ductility Classification')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'ductility_confusion_matrix_1_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print('\nConfusion matrix plot saved to ' + plot_filename)
    clf_ductility.fit(X_ductility, y_ductility)
    model_filename = os.path.join(data_dir, 'ductility_classifier.joblib')
    joblib.dump(clf_ductility, model_filename)
    print('Trained ductility classifier saved to ' + model_filename)

if __name__ == '__main__':
    train_ductility_model()