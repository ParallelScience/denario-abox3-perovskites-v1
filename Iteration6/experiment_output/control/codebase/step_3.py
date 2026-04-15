# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_mechanical_models():
    data_dir = "data/"
    input_path = os.path.join(data_dir, "cleaned_perovskite_data.csv")
    print("Loading dataset from " + input_path + "...")
    df = pd.read_csv(input_path)
    base_features = ['density', 'tau', 'mu', 'en_diff', 'A_radius', 'B_radius', 'formation_energy_per_atom']
    crystal_dummies = pd.get_dummies(df['crystal_system'], prefix='crystal', drop_first=False)
    crystal_dummies = crystal_dummies.astype(float)
    X_all = pd.concat([df[base_features], crystal_dummies], axis=1)
    feature_names = X_all.columns.tolist()
    has_elastic = df['K_VRH'].notnull() & df['G_VRH'].notnull()
    physically_consistent = has_elastic & (df['K_VRH'] > 0) & (df['K_VRH'] < 300) & (df['G_VRH'] > 0)
    df_train = df[physically_consistent]
    X_train = X_all[physically_consistent]
    y_K_train = df_train['K_VRH']
    y_G_train = df_train['G_VRH']
    print("Total samples in dataset: " + str(len(df)))
    print("Total samples with elastic data: " + str(has_elastic.sum()))
    print("Samples after physical consistency filter (0 < K < 300, G > 0): " + str(physically_consistent.sum()))
    print("Number of features used: " + str(X_train.shape[1]))
    print("Features: " + ", ".join(feature_names))
    print("\nTraining GradientBoostingRegressors with Huber loss...")
    gbr_K = GradientBoostingRegressor(loss='huber', random_state=42)
    gbr_G = GradientBoostingRegressor(loss='huber', random_state=42)
    gbr_K.fit(X_train, y_K_train)
    gbr_G.fit(X_train, y_G_train)
    print("Training GaussianProcessRegressors for uncertainty quantification...")
    kernel_K = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    kernel_G = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    gpr_K = Pipeline([('scaler', StandardScaler()), ('gpr', GaussianProcessRegressor(kernel=kernel_K, n_restarts_optimizer=5, random_state=42, normalize_y=True))])
    gpr_G = Pipeline([('scaler', StandardScaler()), ('gpr', GaussianProcessRegressor(kernel=kernel_G, n_restarts_optimizer=5, random_state=42, normalize_y=True))])
    gpr_K.fit(X_train, y_K_train)
    gpr_G.fit(X_train, y_G_train)
    print("Models trained successfully. Generating predictions and uncertainties for the full dataset...")
    K_pred_gbr = gbr_K.predict(X_all)
    G_pred_gbr = gbr_G.predict(X_all)
    K_pred_gpr, K_std_gpr = gpr_K.predict(X_all, return_std=True)
    G_pred_gpr, G_std_gpr = gpr_G.predict(X_all, return_std=True)
    K_var_gpr = K_std_gpr ** 2
    G_var_gpr = G_std_gpr ** 2
    joblib.dump(gbr_K, os.path.join(data_dir, "gbr_K_model.joblib"))
    joblib.dump(gbr_G, os.path.join(data_dir, "gbr_G_model.joblib"))
    joblib.dump(gpr_K, os.path.join(data_dir, "gpr_K_model.joblib"))
    joblib.dump(gpr_G, os.path.join(data_dir, "gpr_G_model.joblib"))
    joblib.dump(feature_names, os.path.join(data_dir, "mechanical_features.joblib"))
    results_df = pd.DataFrame({'material_id': df['material_id'], 'K_VRH_pred_gbr': K_pred_gbr, 'G_VRH_pred_gbr': G_pred_gbr, 'K_VRH_pred_gpr': K_pred_gpr, 'G_VRH_pred_gpr': G_pred_gpr, 'K_VRH_std_gpr': K_std_gpr, 'G_VRH_std_gpr': G_std_gpr, 'K_VRH_var_gpr': K_var_gpr, 'G_VRH_var_gpr': G_var_gpr})
    results_path = os.path.join(data_dir, "mechanical_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print("\nModels saved to " + data_dir)
    print("Predictions and uncertainties saved to " + results_path)
    print("\n--- GBR Predictions Summary (Full Dataset) ---")
    print("K_VRH predicted mean: " + str(round(K_pred_gbr.mean(), 2)) + " GPa, std: " + str(round(K_pred_gbr.std(), 2)) + " GPa")
    print("G_VRH predicted mean: " + str(round(G_pred_gbr.mean(), 2)) + " GPa, std: " + str(round(G_pred_gbr.std(), 2)) + " GPa")
    print("\n--- GPR Uncertainties Summary (Full Dataset) ---")
    print("K_VRH uncertainty (variance) mean: " + str(round(K_var_gpr.mean(), 2)) + " GPa^2, max: " + str(round(K_var_gpr.max(), 2)) + " GPa^2")
    print("G_VRH uncertainty (variance) mean: " + str(round(G_var_gpr.mean(), 2)) + " GPa^2, max: " + str(round(G_var_gpr.max(), 2)) + " GPa^2")
    print("\n--- Model Training R^2 Scores (on filtered subset) ---")
    print("GBR K_VRH R^2: " + str(round(gbr_K.score(X_train, y_K_train), 4)))
    print("GBR G_VRH R^2: " + str(round(gbr_G.score(X_train, y_G_train), 4)))
    print("GPR K_VRH R^2: " + str(round(gpr_K.score(X_train, y_K_train), 4)))
    print("GPR G_VRH R^2: " + str(round(gpr_G.score(X_train, y_G_train), 4)))

if __name__ == '__main__':
    train_mechanical_models()