# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

def main():
    data_dir = "data/"
    input_file = os.path.join(data_dir, "classification_results.csv")
    df = pd.read_csv(input_file)
    features = ['tau', 'mu', 'volume', 'volume_residual', 'en_diff', 'A_Z', 'B_Z', 'A_en', 'B_en', 'A_ie1', 'B_ie1', 'B_valence', 'tilt_proxy']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    train_mask = df['K_VRH'].notnull() & (df['K_VRH'] > 0) & (df['K_VRH'] < 300) & (df['G_VRH'] > -20) & (df['G_VRH'] < 200)
    uncharacterized_mask = df['K_VRH'].isnull()
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    ocsvm.fit(X_scaled[train_mask])
    domain_preds = ocsvm.predict(X_scaled)
    df['in_applicability_domain'] = (domain_preds == 1)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    train_pca = X_pca[train_mask]
    distances = cdist(X_pca, train_pca)
    df['pca_distance_to_boundary'] = distances.min(axis=1)
    print("Applicability Domain and Boundary Analysis\n" + "="*45)
    print("Training set size (elastic subset): " + str(train_mask.sum()))
    print("Uncharacterized materials: " + str(uncharacterized_mask.sum()))
    in_domain_count = (uncharacterized_mask & df['in_applicability_domain']).sum()
    out_domain_count = (uncharacterized_mask & ~df['in_applicability_domain']).sum()
    print("Uncharacterized In-Domain: " + str(in_domain_count))
    print("Uncharacterized Out-of-Domain: " + str(out_domain_count))
    plt.figure(figsize=(10, 8))
    plt.scatter(df.loc[train_mask, 'pca_1'], df.loc[train_mask, 'pca_2'], c='blue', label='Training Set (Elastic Subset)', alpha=0.7, edgecolors='k')
    in_domain_mask = uncharacterized_mask & (df['in_applicability_domain'] == True)
    plt.scatter(df.loc[in_domain_mask, 'pca_1'], df.loc[in_domain_mask, 'pca_2'], c='green', label='Uncharacterized (In-Domain)', alpha=0.6, marker='s', edgecolors='k')
    out_domain_mask = uncharacterized_mask & (df['in_applicability_domain'] == False)
    plt.scatter(df.loc[out_domain_mask, 'pca_1'], df.loc[out_domain_mask, 'pca_2'], c='red', label='Uncharacterized (Out-of-Domain)', alpha=0.6, marker='^', edgecolors='k')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Applicability Domain in PCA Space')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(data_dir, 'applicability_domain_6_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    print("\nPCA plot saved to " + plot_filename)
    ocsvm_filename = os.path.join(data_dir, 'ocsvm_model.joblib')
    pca_filename = os.path.join(data_dir, 'pca_model.joblib')
    scaler_filename = os.path.join(data_dir, 'scaler_model.joblib')
    joblib.dump(ocsvm, ocsvm_filename)
    joblib.dump(pca, pca_filename)
    joblib.dump(scaler, scaler_filename)
    print("Models saved to " + data_dir)
    results_filename = os.path.join(data_dir, 'classification_results.csv')
    df.to_csv(results_filename, index=False)
    print("Updated classification results saved to " + results_filename)

if __name__ == '__main__':
    main()