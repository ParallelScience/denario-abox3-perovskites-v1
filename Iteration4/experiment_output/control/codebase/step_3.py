# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.inspection import permutation_importance

def main():
    data_dir = "data/"
    input_file = os.path.join(data_dir, "classification_results.csv")
    if not os.path.exists(input_file):
        input_file = os.path.join(data_dir, "cleaned_perovskite_data.csv")
    df = pd.read_csv(input_file)
    mask_strict = (df['K_VRH'] > 0) & (df['K_VRH'] < 300) & (df['G_VRH'] > 0) & (df['G_VRH'] < 200)
    df_strict = df[mask_strict].copy()
    df_strict['mechanically_robust'] = (df_strict['G_reuss'] > 0) & (df_strict['K_reuss'] > 0)
    count_strict = df_strict['mechanically_robust'].value_counts().min() if df_strict['mechanically_robust'].nunique() > 1 else 0
    mask_eda = (df['K_VRH'] > 0) & (df['K_VRH'] < 300) & (df['G_VRH'] > -20) & (df['G_VRH'] < 200)
    df_eda = df[mask_eda].copy()
    df_eda['mechanically_robust'] = df_eda['G_VRH'] > 0
    count_eda = df_eda['mechanically_robust'].value_counts().min() if df_eda['mechanically_robust'].nunique() > 1 else 0
    if count_strict >= 5:
        df_elastic = df_strict
        print("Using strict filter: 0 < K_VRH < 300 and 0 < G_VRH < 200.")
    else:
        df_elastic = df_eda
        print("Adjusted filter to include dynamically unstable configurations (-20 < G_VRH < 200) as per EDA report.")
    print("Filtered subset size: " + str(len(df_elastic)) + " samples")
    robust_count = df_elastic['mechanically_robust'].sum()
    unstable_count = len(df_elastic) - robust_count
    print("Class balance:")
    print("Mechanically robust: " + str(robust_count))
    print("Dynamically unstable: " + str(unstable_count))
    features = ['tau', 'mu', 'volume', 'volume_residual', 'en_diff', 'A_Z', 'B_Z', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'B_valence', 'tilt_proxy']
    X = df_elastic[features]
    y = df_elastic['mechanically_robust']
    clf = GradientBoostingClassifier(random_state=42)
    min_class_count = df_elastic['mechanically_robust'].value_counts().min()
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        print("\nError: Not enough samples in the minority class to perform cross-validation.")
        clf.fit(X, y)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = ['accuracy', 'f1', 'roc_auc']
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        print("\nMechanical Viability Classification Results\n" + "="*45)
        print("Cross-validation results (" + str(n_splits) + "-fold stratified):")
        print("Accuracy: " + str(round(scores['test_accuracy'].mean(), 4)) + " +/- " + str(round(scores['test_accuracy'].std(), 4)))
        print("F1 Score: " + str(round(scores['test_f1'].mean(), 4)) + " +/- " + str(round(scores['test_f1'].std(), 4)))
        print("ROC-AUC: " + str(round(scores['test_roc_auc'].mean(), 4)) + " +/- " + str(round(scores['test_roc_auc'].std(), 4)))
        clf.fit(X, y)
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=8)
    importances = result.importances_mean
    std = result.importances_std
    sorted_idx = importances.argsort()
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], xerr=std[sorted_idx], align='center', capsize=4)
    plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
    plt.xlabel('Permutation Feature Importance')
    plt.title('Feature Importances for Mechanical Viability Classifier')
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(data_dir, 'feature_importance_3_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    print("\nFeature importance plot saved to " + plot_filename)
    model_filename = os.path.join(data_dir, 'mechanical_viability_model.joblib')
    joblib.dump(clf, model_filename)
    print("Model saved to " + model_filename)
    df['mechanical_viability_probability'] = clf.predict_proba(df[features])[:, 1]
    df['predicted_mechanically_robust'] = clf.predict(df[features])
    results_filename = os.path.join(data_dir, 'classification_results.csv')
    df.to_csv(results_filename, index=False)
    print("Updated classification results saved to " + results_filename)

if __name__ == '__main__':
    main()