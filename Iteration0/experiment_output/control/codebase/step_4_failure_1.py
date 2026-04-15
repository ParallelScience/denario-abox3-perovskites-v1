# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib

if __name__ == '__main__':
    data_dir = 'data/'
    train_path = os.path.join(data_dir, 'train_dataset.csv')
    test_path = os.path.join(data_dir, 'test_dataset.csv')
    train_df = pd.read_csv(train_path, index_col='material_id')
    test_df = pd.read_csv(test_path, index_col='material_id')
    leakage_cols = ['energy_above_hull', 'formation_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'energy_per_atom']
    mechanical_cols = ['K_VRH', 'K_voigt', 'K_reuss', 'G_VRH', 'G_voigt', 'G_reuss', 'elastic_anisotropy', 'poisson_ratio', 'pugh_ratio']
    string_cols = ['formula', 'chemsys', 'spacegroup_symbol', 'A_site', 'B_site', 'crystal_system', 'magnetic_ordering']
    cols_to_drop = leakage_cols + mechanical_cols + string_cols
    X_train = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns] + ['is_stable'], errors='ignore')
    y_train = train_df['is_stable'].astype(int)
    X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns] + ['is_stable'], errors='ignore')
    y_test = test_df['is_stable'].astype(int)
    X_train = X_train.select_dtypes(include=[np.number, bool]).astype(float)
    X_test = X_test.select_dtypes(include=[np.number, bool]).astype(float)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print('Training set: ' + str(num_neg) + ' unstable (0), ' + str(num_pos) + ' stable (1)')
    print('Calculated scale_pos_weight: ' + str(round(scale_pos_weight, 4)) + '\n')
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='auc', n_jobs=1)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring={'roc_auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}, refit='roc_auc', cv=cv, verbose=0, n_jobs=8)
    print('Starting hyperparameter optimization...')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    cv_results = grid_search.cv_results_
    best_idx = grid_search.best_index_
    print('\nCross-Validation Results (Best Model):')
    print('Best Parameters: ' + str(grid_search.best_params_))
    print('ROC-AUC: ' + str(round(cv_results['mean_test_roc_auc'][best_idx], 4)) + ' ± ' + str(round(cv_results['std_test_roc_auc'][best_idx], 4)))
    print('F1 Score: ' + str(round(cv_results['mean_test_f1'][best_idx], 4)) + ' ± ' + str(round(cv_results['std_test_f1'][best_idx], 4)))
    print('Precision: ' + str(round(cv_results['mean_test_precision'][best_idx], 4)) + ' ± ' + str(round(cv_results['std_test_precision'][best_idx], 4)))
    print('Recall: ' + str(round(cv_results['mean_test_recall'][best_idx], 4)) + ' ± ' + str(round(cv_results['std_test_recall'][best_idx], 4)))
    y_pred = best_model.predict(X_test)
    print('\nTest Set Evaluation:')
    print('Confusion Matrix:')
    print(str(confusion_matrix(y_test, y_pred)))
    print('\nClassification Report:')
    print(str(classification_report(y_test, y_pred, target_names=['Unstable (0)', 'Stable (1)'])))
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print('\nTop 10 Feature Importances:')
    print(importance_df.head(10).to_string(index=False))
    model_path = os.path.join(data_dir, 'stability_xgboost_model.joblib')
    importance_path = os.path.join(data_dir, 'stability_feature_importances.csv')
    joblib.dump(best_model, model_path)
    importance_df.to_csv(importance_path, index=False)
    print('\nModel saved to ' + model_path)
    print('Feature importances saved to ' + importance_path)