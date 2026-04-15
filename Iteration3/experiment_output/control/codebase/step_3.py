# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.impute import SimpleImputer
import joblib
import os

def load_and_filter_data(filepath):
    df = pd.read_csv(filepath)
    elastic_df = df.dropna(subset=['K_VRH', 'G_VRH']).copy()
    initial_count = len(elastic_df)
    elastic_df = elastic_df[(elastic_df['K_VRH'] <= 300) & (elastic_df['G_VRH'] >= 0) & (elastic_df['G_VRH'] <= 200)]
    retained_count = len(elastic_df)
    return elastic_df, initial_count, retained_count

def train_and_evaluate_gpr(X, y, target_name):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'r2': 'r2', 'neg_rmse': 'neg_root_mean_squared_error'}
    cv_results = cross_validate(gpr, X, y, cv=kf, scoring=scoring, return_train_score=False)
    r2 = np.mean(cv_results['test_r2'])
    rmse = -np.mean(cv_results['test_neg_rmse'])
    print(target_name + ' - R^2: ' + str(round(r2, 4)) + ', RMSE: ' + str(round(rmse, 4)) + ' GPa')
    gpr.fit(X, y)
    return gpr, r2, rmse

def main():
    data_dir = 'data/'
    input_filepath = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    elastic_df, initial_count, retained_count = load_and_filter_data(input_filepath)
    print('Initial elastic samples: ' + str(initial_count))
    print('Samples retained after filtering outliers: ' + str(retained_count))
    features = ['nsites', 'volume', 'density', 'density_atomic', 'spacegroup_number', 'A_Z', 'A_radius', 'A_en', 'A_ie1', 'A_group', 'B_Z', 'B_radius', 'B_en', 'B_ie1', 'B_group', 'B_valence', 'tau', 'mu', 'en_diff', 'VEC', 'volume_residual']
    X = elastic_df[features]
    y_K = elastic_df['K_VRH']
    y_G = elastic_df['G_VRH']
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print('\n--- Cross-Validation Results (5-Fold) ---')
    gpr_K, r2_K, rmse_K = train_and_evaluate_gpr(X_scaled, y_K, 'K_VRH')
    gpr_G, r2_G, rmse_G = train_and_evaluate_gpr(X_scaled, y_G, 'G_VRH')
    joblib.dump(gpr_K, os.path.join(data_dir, 'gpr_K_model.joblib'))
    joblib.dump(gpr_G, os.path.join(data_dir, 'gpr_G_model.joblib'))
    joblib.dump(scaler, os.path.join(data_dir, 'gpr_scaler.joblib'))
    joblib.dump(imputer, os.path.join(data_dir, 'gpr_imputer.joblib'))
    metrics_df = pd.DataFrame({'Target': ['K_VRH', 'G_VRH'], 'CV_R2': [r2_K, r2_G], 'CV_RMSE': [rmse_K, rmse_G]})
    metrics_df.to_csv(os.path.join(data_dir, 'gpr_cv_metrics.csv'), index=False)
    print('\nModels and metrics saved successfully.')

if __name__ == '__main__':
    main()