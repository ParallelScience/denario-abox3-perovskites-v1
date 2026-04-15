# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    filepath = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(filepath)
    df_elastic = df.dropna(subset=['K_VRH', 'G_VRH']).copy()
    print('Original elastic subset size: ' + str(len(df_elastic)))
    mask = (df_elastic['K_VRH'] <= 300) & (df_elastic['K_VRH'] > 0) & (df_elastic['G_VRH'] >= 0)
    df_elastic_filtered = df_elastic[mask].copy()
    print('Filtered elastic subset size: ' + str(len(df_elastic_filtered)))
    base_features = ['density', 'A_Z', 'B_Z', 'A_en', 'B_en', 'A_radius', 'B_radius', 'tau', 'mu', 'VEC', 'en_diff']
    crystal_system_cols = [col for col in df.columns if col.startswith('crystal_system_')]
    features = base_features + crystal_system_cols
    X_elastic = df_elastic_filtered[features]
    y_K = df_elastic_filtered['K_VRH']
    y_G = df_elastic_filtered['G_VRH']
    X_train, X_test, y_K_train, y_K_test, y_G_train, y_G_test = train_test_split(X_elastic, y_K, y_G, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)
    gpr_K = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=True)
    gpr_G = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=True)
    print('Training GPR for K_VRH...')
    gpr_K.fit(X_train_scaled, y_K_train)
    print('Training GPR for G_VRH...')
    gpr_G.fit(X_train_scaled, y_G_train)
    K_pred, K_std = gpr_K.predict(X_test_scaled, return_std=True)
    G_pred, G_std = gpr_G.predict(X_test_scaled, return_std=True)
    print('K_VRH Test R2: ' + str(round(r2_score(y_K_test, K_pred), 4)))
    print('K_VRH Test RMSE: ' + str(round(np.sqrt(mean_squared_error(y_K_test, K_pred)), 4)) + ' GPa')
    print('G_VRH Test R2: ' + str(round(r2_score(y_G_test, G_pred), 4)))
    print('G_VRH Test RMSE: ' + str(round(np.sqrt(mean_squared_error(y_G_test, G_pred)), 4)) + ' GPa')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.errorbar(y_K_test, K_pred, yerr=K_std, fmt='o', ecolor='lightgray', elinewidth=2, capsize=0, alpha=0.7, markerfacecolor='blue', markeredgecolor='blue')
    max_K = max(y_K_test.max(), K_pred.max()) + 20
    ax1.plot([0, max_K], [0, max_K], 'k--', lw=2)
    ax1.set_xlabel('Actual K_VRH (GPa)')
    ax1.set_ylabel('Predicted K_VRH (GPa)')
    ax1.set_title('GPR Predictions for K_VRH')
    ax1.set_xlim(0, max_K)
    ax1.set_ylim(0, max_K)
    ax2.errorbar(y_G_test, G_pred, yerr=G_std, fmt='o', ecolor='lightgray', elinewidth=2, capsize=0, alpha=0.7, markerfacecolor='green', markeredgecolor='green')
    max_G = max(y_G_test.max(), G_pred.max()) + 20
    ax2.plot([0, max_G], [0, max_G], 'k--', lw=2)
    ax2.set_xlabel('Actual G_VRH (GPa)')
    ax2.set_ylabel('Predicted G_VRH (GPa)')
    ax2.set_title('GPR Predictions for G_VRH')
    ax2.set_xlim(0, max_G)
    ax2.set_ylim(0, max_G)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'gpr_diagnostic_1_' + timestamp + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print('Diagnostic plot saved to ' + plot_filepath)
    df_uncharacterized = df[df['K_VRH'].isna()].copy()
    print('Number of uncharacterized materials: ' + str(len(df_uncharacterized)))
    X_uncharacterized = df_uncharacterized[features]
    X_uncharacterized_scaled = scaler.transform(X_uncharacterized)
    K_pred_all, K_std_all = gpr_K.predict(X_uncharacterized_scaled, return_std=True)
    G_pred_all, G_std_all = gpr_G.predict(X_uncharacterized_scaled, return_std=True)
    K_pred_safe = np.clip(K_pred_all, a_min=1e-3, a_max=None)
    Pugh_pred = G_pred_all / K_pred_safe
    Pugh_var = (1 / K_pred_safe**2) * (G_std_all**2) + (G_pred_all**2 / K_pred_safe**4) * (K_std_all**2)
    Pugh_std = np.sqrt(Pugh_var)
    df['predicted_K_VRH'] = np.nan
    df['predicted_K_VRH_std'] = np.nan
    df['predicted_G_VRH'] = np.nan
    df['predicted_G_VRH_std'] = np.nan
    df['predicted_pugh_ratio'] = np.nan
    df['predicted_pugh_ratio_std'] = np.nan
    uncharacterized_indices = df_uncharacterized.index
    df.loc[uncharacterized_indices, 'predicted_K_VRH'] = K_pred_all
    df.loc[uncharacterized_indices, 'predicted_K_VRH_std'] = K_std_all
    df.loc[uncharacterized_indices, 'predicted_G_VRH'] = G_pred_all
    df.loc[uncharacterized_indices, 'predicted_G_VRH_std'] = G_std_all
    df.loc[uncharacterized_indices, 'predicted_pugh_ratio'] = Pugh_pred
    df.loc[uncharacterized_indices, 'predicted_pugh_ratio_std'] = Pugh_std
    df.to_csv(filepath, index=False)
    print('Dataset with GPR predictions saved to ' + filepath)

if __name__ == '__main__':
    main()