# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import joblib
import os

if __name__ == '__main__':
    data_dir = 'data/'
    full_data_path = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    full_df = pd.read_csv(full_data_path, index_col='material_id')
    uncharacterized_df = full_df[full_df['K_VRH'].isna()].copy()
    categorical_cols = ['crystal_system', 'magnetic_ordering']
    full_df_encoded = pd.get_dummies(full_df, columns=categorical_cols, drop_first=False, dtype=int)
    uncharacterized_encoded = full_df_encoded.loc[uncharacterized_df.index].copy()
    stability_model = joblib.load(os.path.join(data_dir, 'stability_xgboost_model.joblib'))
    rf_K = joblib.load(os.path.join(data_dir, 'rf_K_VRH_model.joblib'))
    rf_G = joblib.load(os.path.join(data_dir, 'rf_G_VRH_model.joblib'))
    mech_features = joblib.load(os.path.join(data_dir, 'mechanical_features.joblib'))
    if hasattr(stability_model, 'feature_names_in_'):
        stability_features = list(stability_model.feature_names_in_)
    elif hasattr(stability_model, 'get_booster'):
        stability_features = stability_model.get_booster().feature_names
    else:
        leakage_cols = ['energy_above_hull', 'formation_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'energy_per_atom', 'is_stable']
        mechanical_cols = ['K_VRH', 'K_voigt', 'K_reuss', 'G_VRH', 'G_voigt', 'G_reuss', 'elastic_anisotropy', 'poisson_ratio', 'pugh_ratio']
        string_cols = ['formula', 'chemsys', 'spacegroup_symbol', 'A_site', 'B_site', 'crystal_system', 'magnetic_ordering']
        cols_to_drop = leakage_cols + mechanical_cols + string_cols
        temp_X = uncharacterized_encoded.drop(columns=[c for c in cols_to_drop if c in uncharacterized_encoded.columns], errors='ignore')
        temp_X = temp_X.select_dtypes(include=[np.number, bool]).astype(float)
        stability_features = list(temp_X.columns)
    for col in stability_features:
        if col not in uncharacterized_encoded.columns:
            uncharacterized_encoded[col] = 0
    X_stab = uncharacterized_encoded[stability_features].select_dtypes(include=[np.number, bool]).astype(float)
    stability_probs = stability_model.predict_proba(X_stab)[:, 1]
    for col in mech_features:
        if col not in uncharacterized_encoded.columns:
            uncharacterized_encoded[col] = 0
    X_mech = uncharacterized_encoded[mech_features].select_dtypes(include=[np.number, bool]).astype(float)
    K_pred = rf_K.predict(X_mech)
    G_pred = rf_G.predict(X_mech)
    results_df = uncharacterized_df[['formula']].copy()
    results_df['stability_prob'] = stability_probs
    results_df['K_VRH_pred'] = K_pred
    results_df['G_VRH_pred'] = G_pred
    results_df['mechanical_viability'] = (results_df['K_VRH_pred'] > 50) & (results_df['K_VRH_pred'] < 250) & (results_df['G_VRH_pred'] > 0)
    num_viable = results_df['mechanical_viability'].sum()
    num_stable = (results_df['stability_prob'] > 0.5).sum()
    print('Total uncharacterized materials: ' + str(len(results_df)))
    print('Materials passing mechanical viability threshold: ' + str(num_viable))
    print('Materials with stability probability > 0.5: ' + str(num_stable))
    results_df['final_score'] = results_df['stability_prob'] * results_df['mechanical_viability'].astype(int)
    top_20 = results_df.sort_values(by=['final_score', 'stability_prob'], ascending=[False, False]).head(20)
    print('\nTop-20 Ranked Candidates:')
    print(top_20[['formula', 'stability_prob', 'K_VRH_pred', 'G_VRH_pred', 'final_score']].to_string())
    output_path = os.path.join(data_dir, 'pipeline_predictions.csv')
    results_df.to_csv(output_path)
    print('\nFinal predictions saved to ' + output_path)