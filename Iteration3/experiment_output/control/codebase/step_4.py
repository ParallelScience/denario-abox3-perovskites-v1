# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import MinMaxScaler

def main():
    data_dir = 'data/'
    input_filepath = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(input_filepath)
    features = ['nsites', 'volume', 'density', 'density_atomic', 'spacegroup_number', 'A_Z', 'A_radius', 'A_en', 'A_ie1', 'A_group', 'B_Z', 'B_radius', 'B_en', 'B_ie1', 'B_group', 'B_valence', 'tau', 'mu', 'en_diff', 'VEC', 'volume_residual']
    X_all = df[features]
    y_all = df['is_stable_soft'].astype(int)
    imputer_gbc = SimpleImputer(strategy='median')
    X_all_imputed = imputer_gbc.fit_transform(X_all)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_all)
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_all_imputed, y_all, sample_weight=sample_weights)
    df['prob_stable'] = gbc.predict_proba(X_all_imputed)[:, 1]
    elastic_mask = df['K_VRH'].notnull() & df['G_VRH'].notnull()
    df_elastic = df[elastic_mask].copy()
    df_unchar = df[~elastic_mask].copy()
    gpr_K = joblib.load(os.path.join(data_dir, 'gpr_K_model.joblib'))
    gpr_G = joblib.load(os.path.join(data_dir, 'gpr_G_model.joblib'))
    scaler = joblib.load(os.path.join(data_dir, 'gpr_scaler.joblib'))
    imputer_gpr = joblib.load(os.path.join(data_dir, 'gpr_imputer.joblib'))
    X_elastic = df_elastic[features]
    X_elastic_imputed = imputer_gpr.transform(X_elastic)
    X_elastic_scaled = scaler.transform(X_elastic_imputed)
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    ocsvm.fit(X_elastic_scaled)
    X_unchar = df_unchar[features]
    X_unchar_imputed = imputer_gpr.transform(X_unchar)
    X_unchar_scaled = scaler.transform(X_unchar_imputed)
    domain_preds = ocsvm.predict(X_unchar_scaled)
    df_unchar['in_domain'] = domain_preds == 1
    df_unchar['domain_status'] = np.where(df_unchar['in_domain'], 'In-Domain', 'Out-of-Domain (High-Uncertainty)')
    in_domain_count = df_unchar['in_domain'].sum()
    out_domain_count = len(df_unchar) - in_domain_count
    print('--- Applicability Domain Classification ---')
    print('Total uncharacterized materials: ' + str(len(df_unchar)))
    print('In-Domain: ' + str(in_domain_count) + ' (' + str(round(in_domain_count/len(df_unchar)*100, 1)) + '%)')
    print('Out-of-Domain: ' + str(out_domain_count) + ' (' + str(round(out_domain_count/len(df_unchar)*100, 1)) + '%)\n')
    K_pred, K_std = gpr_K.predict(X_unchar_scaled, return_std=True)
    G_pred, G_std = gpr_G.predict(X_unchar_scaled, return_std=True)
    df_unchar['K_VRH_pred'] = K_pred
    df_unchar['K_VRH_std'] = K_std
    df_unchar['G_VRH_pred'] = G_pred
    df_unchar['G_VRH_std'] = G_std
    unc_scaler = MinMaxScaler()
    total_std = K_std + G_std
    normalized_unc = unc_scaler.fit_transform(total_std.reshape(-1, 1)).flatten()
    df_unchar['UCB_score'] = df_unchar['prob_stable'] + normalized_unc
    df_ranked = df_unchar.sort_values(by='UCB_score', ascending=False)
    output_path = os.path.join(data_dir, 'ranked_candidates.csv')
    df_ranked.to_csv(output_path, index=False)
    print('Ranked candidate list saved to: ' + output_path + '\n')
    print('--- Top 50 Candidates for Active Learning (Ranked by UCB Score) ---')
    cols_to_print = ['material_id', 'formula', 'spacegroup_symbol', 'prob_stable', 'UCB_score', 'domain_status', 'K_VRH_pred', 'K_VRH_std', 'G_VRH_pred', 'G_VRH_std', 'energy_above_hull', 'band_gap', 'is_metal']
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(df_ranked[cols_to_print].head(50).to_string(index=False))
    joblib.dump(ocsvm, os.path.join(data_dir, 'ocsvm_model.joblib'))
    df.to_csv(os.path.join(data_dir, 'cleaned_perovskite_data_with_probs.csv'), index=False)

if __name__ == '__main__':
    main()