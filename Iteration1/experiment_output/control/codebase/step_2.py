# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate, KFold
import time
import os

def train_stability_model():
    data_path = 'data/cleaned_perovskite_data.csv'
    df = pd.read_csv(data_path, index_col='material_id')
    features = ['A_Z', 'B_Z', 'A_radius', 'B_radius', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'A_group', 'B_group', 'en_diff', 'tau', 'mu', 'B_valence', 'log_volume', 'abs_tau_diff', 'radius_diff', 'ie_ratio']
    X = df[features]
    y = np.log1p(df['energy_above_hull'])
    model = GradientBoostingRegressor(random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'r2': 'r2', 'neg_rmse': 'neg_root_mean_squared_error', 'neg_mae': 'neg_mean_absolute_error'}
    cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
    r2_mean = np.mean(cv_results['test_r2'])
    r2_std = np.std(cv_results['test_r2'])
    rmse_mean = -np.mean(cv_results['test_neg_rmse'])
    rmse_std = np.std(cv_results['test_neg_rmse'])
    mae_mean = -np.mean(cv_results['test_neg_mae'])
    mae_std = np.std(cv_results['test_neg_mae'])
    print('========================================')
    print('--- CROSS-VALIDATION METRICS (5-Fold) ---')
    print('========================================')
    print('Target: log1p(energy_above_hull)')
    print('R2 Score: ' + str(round(r2_mean, 4)) + ' +/- ' + str(round(r2_std, 4)))
    print('RMSE:     ' + str(round(rmse_mean, 4)) + ' +/- ' + str(round(rmse_std, 4)))
    print('MAE:      ' + str(round(mae_mean, 4)) + ' +/- ' + str(round(mae_std, 4)))
    model.fit(X, y)
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print('\n========================================')
    print('--- FEATURE IMPORTANCES ---')
    print('========================================')
    print(feat_imp_df.to_string(index=False))
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1], color='skyblue')
    plt.xlabel('Feature Importance (dimensionless)')
    plt.title('Gradient Boosting Regressor - Feature Importances')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join('data', 'feature_importance_stability_1_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print('\nFeature importance plot saved to ' + plot_filename)
    model_filename = os.path.join('data', 'stability_model.joblib')
    joblib.dump(model, model_filename)
    print('Trained model saved to ' + model_filename)

if __name__ == '__main__':
    train_stability_model()