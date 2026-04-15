# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np

def main():
    data_dir = "data/"
    input_file = os.path.join(data_dir, "classification_results.csv")
    df = pd.read_csv(input_file)
    uncharacterized_mask = df['K_VRH'].isnull()
    uncharacterized = df[uncharacterized_mask].copy()
    print("Candidate Selection and Ranking\n" + "="*45)
    print("Total uncharacterized materials: " + str(len(uncharacterized)))
    rank_stab = uncharacterized['stability_probability'].rank(ascending=False)
    rank_mech = uncharacterized['mechanical_viability_probability'].rank(ascending=False)
    rank_dist = uncharacterized['pca_distance_to_boundary'].rank(ascending=True)
    uncharacterized['composite_rank_score'] = rank_stab + rank_mech + rank_dist
    uncharacterized = uncharacterized.sort_values('composite_rank_score', ascending=True)
    uncharacterized['cluster_id'] = pd.factorize(uncharacterized['chemsys'])[0]
    full_ranked_file = os.path.join(data_dir, 'full_ranked_uncharacterized_materials.csv')
    uncharacterized.to_csv(full_ranked_file, index=False)
    print("Full ranked list saved to " + full_ranked_file)
    non_metallic = uncharacterized[uncharacterized['predicted_is_metal'] == False].copy()
    print("Non-metallic uncharacterized materials: " + str(len(non_metallic)))
    diverse_candidates = non_metallic.drop_duplicates(subset=['chemsys'], keep='first')
    print("Diverse non-metallic candidates (unique chemsys): " + str(len(diverse_candidates)))
    top_20 = diverse_candidates.head(20).copy()
    cols_to_keep = ['material_id', 'formula', 'chemsys', 'stability_probability', 'mechanical_viability_probability', 'pca_distance_to_boundary', 'predicted_band_gap', 'cluster_id']
    top_20 = top_20[cols_to_keep]
    print("\nTop-20 Ranked Candidates:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(top_20.to_string(index=False))
    top_20_file = os.path.join(data_dir, 'top_20_candidates.csv')
    top_20.to_csv(top_20_file, index=False)
    print("\nTop-20 candidates saved to " + top_20_file)

if __name__ == '__main__':
    main()