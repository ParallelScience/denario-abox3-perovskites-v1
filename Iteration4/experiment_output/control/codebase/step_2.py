# filename: codebase/step_2.py
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

def train_and_evaluate_stability_model(df):
    df['is_stable'] = df['energy_above_hull'] == 0
    features = ['tau', 'mu', 'volume', 'volume_residual', 'en_diff', 'A_Z', 'B_Z', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'B_valence', 'VEC', 'tilt_proxy']
    X = df[features]
    y = df['is_stable']
    clf = GradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['f1', 'precision', 'recall', 'roc_auc']
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
    print('Thermodynamic Stability Classification Results\n' + '='*46)
    print('Cross-validation results (5-fold stratified):')
    print('F1: ' + str(round(scores['test_f1'].mean(), 4)) + ' +/- ' + str(round(scores['test_f1'].std(), 4)))
    print('Precision: ' + str(round(scores['test_precision'].mean(), 4)) + ' +/- ' + str(round(scores['test_precision'].std(), 4)))
    print('Recall: ' + str(round(scores['test_recall'].mean(), 4)) + ' +/- ' + str(round(scores['test_recall'].std(), 4)))
    print('ROC-AUC: ' + str(round(scores['test_roc_auc'].mean(), 4)) + ' +/- ' + str(round(scores['test_roc_auc'].std(), 4)))
    clf.fit(X, y)
    df['predicted_is_stable'] = clf.predict(X)
    df['stability_probability'] = clf.predict_proba(X)[:, 1]
    TP = (y == True) & (df['predicted_is_stable'] == True)
    FP = (y == False) & (df['predicted_is_stable'] == True)
    TN = (y == False) & (df['predicted_is_stable'] == False)
    FN = (y == True) & (df['predicted_is_stable'] == False)
    print('\nClassification counts on full dataset:')
    print('True Positives: ' + str(TP.sum()))
    print('False Positives: ' + str(FP.sum()))
    print('True Negatives: ' + str(TN.sum()))
    print('False Negatives: ' + str(FN.sum()))
    masks = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return clf, df, masks

def plot_reaction_energy_distribution(df, masks):
    plt.figure(figsize=(10, 6))
    colors = {'True Positives': 'blue', 'False Positives': 'red', 'True Negatives': 'green'}
    for mask_key, label in [('TP', 'True Positives'), ('FP', 'False Positives'), ('TN', 'True Negatives')]:
        data = df.loc[masks[mask_key], 'equilibrium_reaction_energy_per_atom'].dropna()
        if len(data) > 0:
            plt.hist(data, bins=15, alpha=0.5, label=label + ' (n=' + str(len(data)) + ')', color=colors[label], edgecolor='black')
        else:
            plt.hist([], alpha=0.5, label=label + ' (n=0)', color=colors[label])
    plt.xlabel('Equilibrium Reaction Energy per Atom (eV/atom)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Equilibrium Reaction Energy per Atom')
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join('data', 'eq_reaction_energy_dist_2_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('\nPlot saved to ' + plot_filename)

def print_top_false_positives(df, masks):
    fp_df = df[masks['FP']].copy()
    fp_df = fp_df.sort_values(by='stability_probability', ascending=False).head(20)
    cols_to_print = ['formula', 'energy_above_hull', 'equilibrium_reaction_energy_per_atom', 'stability_probability']
    print('\nTop-20 False Positive Materials:')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(fp_df[cols_to_print].to_string(index=False))

def main():
    input_file = os.path.join('data', 'cleaned_perovskite_data.csv')
    df = pd.read_csv(input_file)
    clf, df, masks = train_and_evaluate_stability_model(df)
    plot_reaction_energy_distribution(df, masks)
    print_top_false_positives(df, masks)
    model_filename = os.path.join('data', 'stability_model.joblib')
    results_filename = os.path.join('data', 'classification_results.csv')
    joblib.dump(clf, model_filename)
    df.to_csv(results_filename, index=False)
    print('\nModel saved to ' + model_filename)
    print('Classification results saved to ' + results_filename)

if __name__ == '__main__':
    main()