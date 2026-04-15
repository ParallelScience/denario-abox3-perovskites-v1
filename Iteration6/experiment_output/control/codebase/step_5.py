# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

def train_mechanical_viability_classifier():
    data_dir = "data/"
    input_path = os.path.join(data_dir, "cleaned_perovskite_data.csv")
    print("Loading dataset from " + input_path + "...")
    df = pd.read_csv(input_path)
    base_features = ['tau_strain', 'mu_strain', 'tau', 'mu', 'en_diff', 'A_radius', 'B_radius', 'density', 'formation_energy_per_atom', 'log_volume']
    crystal_dummies = pd.get_dummies(df['crystal_system'], prefix='crystal', drop_first=False).astype(float)
    glazer_dummies = pd.get_dummies(df['glazer_tilt'], prefix='tilt', drop_first=False).astype(float)
    X_full = pd.concat([df[base_features], crystal_dummies, glazer_dummies], axis=1)
    has_elastic = df['K_VRH'].notnull() & df['G_VRH'].notnull()
    df_elastic = df[has_elastic].copy()
    print("Total samples with elastic data: " + str(len(df_elastic)))
    df_elastic['is_mechanically_viable'] = ((df_elastic['K_VRH'] > 0) & (df_elastic['K_VRH'] < 300) & (df_elastic['G_VRH'] > 0)).astype(int)
    n_viable = df_elastic['is_mechanically_viable'].sum()
    n_total = len(df_elastic)
    print("Mechanically viable samples: " + str(n_viable) + " / " + str(n_total) + " (" + str(round(n_viable/n_total*100, 1)) + "%)")
    X = X_full[has_elastic].copy()
    y = df_elastic['is_mechanically_viable']
    feature_names = X.columns.tolist()
    print("Number of features: " + str(X.shape[1]))
    clf = GradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring)
    print("\n--- 5-Fold Cross-Validation Results ---")
    print("Accuracy:  " + str(round(np.mean(cv_results['test_accuracy']), 4)) + " ± " + str(round(np.std(cv_results['test_accuracy']), 4)))
    print("ROC-AUC:   " + str(round(np.mean(cv_results['test_roc_auc']), 4)) + " ± " + str(round(np.std(cv_results['test_roc_auc']), 4)))
    print("F1 Score:  " + str(round(np.mean(cv_results['test_f1']), 4)) + " ± " + str(round(np.std(cv_results['test_f1']), 4)))
    print("Precision: " + str(round(np.mean(cv_results['test_precision']), 4)) + " ± " + str(round(np.std(cv_results['test_precision']), 4)))
    print("Recall:    " + str(round(np.mean(cv_results['test_recall']), 4)) + " ± " + str(round(np.std(cv_results['test_recall']), 4)))
    print("\nTraining final classifier on all 215 samples...")
    clf.fit(X, y)
    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\n--- Top 10 Feature Importances ---")
    for idx, row in feat_imp_df.head(10).iterrows():
        print(row['Feature'] + ": " + str(round(row['Importance'], 4)))
    model_path = os.path.join(data_dir, "mechanical_viability_classifier.joblib")
    joblib.dump(clf, model_path)
    features_path = os.path.join(data_dir, "mechanical_viability_features.joblib")
    joblib.dump(feature_names, features_path)
    importances_path = os.path.join(data_dir, "mechanical_viability_importances.csv")
    feat_imp_df.to_csv(importances_path, index=False)
    print("\nModel saved to " + model_path)
    print("Feature names saved to " + features_path)
    print("Feature importances saved to " + importances_path)

if __name__ == '__main__':
    train_mechanical_viability_classifier()