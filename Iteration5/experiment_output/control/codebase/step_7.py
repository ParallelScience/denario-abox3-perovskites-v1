# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    filepath = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    print('Loading dataset...')
    df = pd.read_csv(filepath)
    print('Re-training GPR models for mechanical properties...')
    df_elastic = df.dropna(subset=['K_VRH', 'G_VRH']).copy()
    mask = (df_elastic['K_VRH'] <= 300) & (df_elastic['K_VRH'] > 0) & (df_elastic['G_VRH'] >= 0)
    df_elastic_filtered = df_elastic[mask].copy()
    base_features = ['density', 'A_Z', 'B_Z', 'A_en', 'B_en', 'A_radius', 'B_radius', 'tau', 'mu', 'VEC', 'en_diff']
    crystal_system_cols = [col for col in df.columns if col.startswith('crystal_system_')]
    features = base_features + crystal_system_cols
    X_elastic = df_elastic_filtered[features]
    y_K = df_elastic_filtered['K_VRH']
    y_G = df_elastic_filtered['G_VRH']
    X_train, X_test, y_K_train, y_K_test, y_G_train, y_G_test = train_test_split(X_elastic, y_K, y_G, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)
    gpr_K = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=True)
    gpr_G = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=True)
    gpr_K.fit(X_train_scaled, y_K_train)
    gpr_G.fit(X_train_scaled, y_G_train)
    print('Identifying top-10 Pareto-optimal candidates...')
    df_uncharacterized = df[df['K_VRH'].isna()].copy()
    df_uncharacterized = df_uncharacterized.dropna(subset=['predicted_is_stable_prob', 'predicted_pugh_ratio_std'])
    df_uncharacterized = df_uncharacterized.reset_index(drop=True)
    obj1 = df_uncharacterized['predicted_is_stable_prob'].values
    obj2 = df_uncharacterized['predicted_pugh_ratio_std'].values
    n = len(obj1)
    pareto_front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (obj1[j] >= obj1[i] and obj2[j] <= obj2[i]) and (obj1[j] > obj1[i] or obj2[j] < obj2[i]):
                dominated = True
                break
        if not dominated:
            pareto_front.append(i)
    pareto_front = np.array(pareto_front)
    pareto_sorted_indices = pareto_front[np.argsort(obj1[pareto_front])[::-1]]
    top_10_indices = pareto_sorted_indices[:10]
    print('Performing sensitivity analysis (1000 perturbations per candidate)...')
    features_to_perturb = ['A_radius', 'B_radius', 'tau', 'mu']
    feature_indices = [features.index(f) for f in features_to_perturb]
    np.random.seed(42)
    n_samples = 1000
    results = []
    plot_data = []
    for idx in top_10_indices:
        row = df_uncharacterized.iloc[idx]
        formula = row['formula']
        x_orig = row[features].values.astype(float)
        X_perturbed = np.tile(x_orig, (n_samples, 1))
        perturbation_factors = np.random.uniform(0.95, 1.05, size=(n_samples, len(features_to_perturb)))
        for i, f_idx in enumerate(feature_indices):
            X_perturbed[:, f_idx] *= perturbation_factors[:, i]
        X_perturbed_scaled = scaler.transform(X_perturbed)
        K_pred = gpr_K.predict(X_perturbed_scaled)
        G_pred = gpr_G.predict(X_perturbed_scaled)
        K_pred_safe = np.clip(K_pred, a_min=1e-3, a_max=None)
        pugh_ratios = G_pred / K_pred_safe
        plot_data.append((formula, pugh_ratios))
        ductile_frac = np.mean(pugh_ratios < 0.571)
        brittle_frac = np.mean(pugh_ratios > 0.571)
        if ductile_frac >= 0.95:
            classification = 'Robustly Ductile'
        elif brittle_frac >= 0.95:
            classification = 'Robustly Brittle'
        else:
            classification = 'Ambiguous'
        results.append({'Formula': formula, 'Mean Pugh': np.mean(pugh_ratios), 'Std Pugh': np.std(pugh_ratios), 'Ductile %': ductile_frac * 100, 'Brittle %': brittle_frac * 100, 'Classification': classification})
    print('\nSensitivity Analysis Results:')
    print('-' * 100)
    print('Formula         | Mean G/K   | Std G/K    | Ductile %  | Brittle %  | Classification      ')
    print('-' * 100)
    for res in results:
        form = str(res['Formula']).ljust(15)
        mean_p = str(round(res['Mean Pugh'], 4)).ljust(10)
        std_p = str(round(res['Std Pugh'], 4)).ljust(10)
        duct = str(round(res['Ductile %'], 1)).ljust(10)
        brit = str(round(res['Brittle %'], 1)).ljust(10)
        cls = str(res['Classification']).ljust(20)
        print(form + ' | ' + mean_p + ' | ' + std_p + ' | ' + duct + ' | ' + brit + ' | ' + cls)
    print('-' * 100 + '\n')
    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = [d[1] for d in plot_data]
    labels = [d[0] for d in plot_data]
    ax.boxplot(data_to_plot, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'), medianprops=dict(color='red', linewidth=2))
    ax.axhline(y=0.571, color='r', linestyle='--', label='Ductile/Brittle Threshold (G/K = 0.571)')
    ax.set_ylabel('Predicted Pugh Ratio (G/K)')
    ax.set_title('Sensitivity of Pugh Ratio to +/- 5% Perturbations in Structural Features')
    ax.legend()
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'sensitivity_analysis_1_' + timestamp + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print('Sensitivity analysis plot saved to ' + plot_filepath)

if __name__ == '__main__':
    main()