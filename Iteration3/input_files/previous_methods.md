1. **Data Preprocessing and Feature Engineering**
   - Remove columns: `surface_energy`, `work_function`, `A_period`, `B_period`, `theoretical`, and any duplicate columns. Set `material_id` as the index.
   - Apply log-transformation to `volume` and `elastic_anisotropy`.
   - Engineer features: `abs_tau_diff` = `abs(tau - 1.0)`, `radius_diff` = `A_radius - B_radius`, `en_diff` = `abs(B_en - A_en)`, `en_var` = variance of electronegativities, and `VEC` (total valence electron count).
   - Drop `radius_ratio` to mitigate multicollinearity with `tau`.
   - Retain only compositional and structural descriptors to ensure no leakage from DFT-calculated properties.

2. **Thermodynamic Stability Classification**
   - Define binary target: `is_stable` (True if `energy_above_hull` == 0, False otherwise). Optionally, define a "near-stable" class (e.g., `energy_above_hull` < 0.05 eV/atom) for sensitivity analysis.
   - Train a Gradient Boosted Classifier to predict `is_stable`.
   - Perform Feature Importance Sensitivity Analysis: compare model performance (AUC-ROC) with and without `volume`. If performance is stable without `volume`, prioritize compositional/bonding features to avoid "geometric proxy" bias.

3. **Mechanical Viability Classification**
   - Define "Mechanically Viable" class using percentile-based clipping (1st to 99th percentile) of the 215-sample elastic subset to exclude pathological outliers (e.g., negative moduli or unphysical stiffness).
   - Address class imbalance using SMOTE or class-weight balancing.
   - Train a binary classifier (starting with a regularized Logistic Regression baseline, then Gradient Boosted trees) to distinguish between "Mechanically Viable" and "Unstable/Pathological" configurations.

4. **Generalization Validation**
   - Implement "Leave-One-Chemical-System-Out" (LOCO) cross-validation, defining a "chemical system" as all materials sharing the same A-site or B-site element.
   - Evaluate model performance on unseen chemical systems to ensure the model generalizes to new chemistry rather than memorizing specific element combinations.

5. **Pareto-Optimal Frontier Analysis**
   - Identify the Pareto-optimal frontier in the 3D space of (Predicted Stability Probability, Predicted Mechanical Viability Probability, and Predicted Ductility Probability).
   - Use probability outputs rather than binary labels for a granular ranking.
   - Apply a diversity constraint: cluster Pareto-optimal candidates by chemical system and select the top-performing representative from each cluster to ensure a diverse set of recommendations.

6. **Ductility and Electronic Profiling**
   - Stage 1: Train a classifier to predict `is_metal` (evaluate using Precision-Recall curves).
   - Stage 2: Train a regressor on the non-metallic subset to predict `band_gap` (evaluate using Mean Absolute Error).
   - Predict "Ductile" vs. "Brittle" category using a classifier trained on the `pugh_ratio` of the viable subset.

7. **Sensitivity and Bias Mitigation**
   - Calculate the Mahalanobis distance of top-ranked candidates from the training set distribution.
   - Define a "Novelty Threshold" (e.g., > 3 standard deviations from the training mean) to flag "High-Risk/High-Reward" candidates that represent genuine extrapolations.

8. **Final Candidate Mapping**
   - Generate the final list of candidates residing on the Pareto-optimal frontier.
   - Export the final dataset containing `material_id`, stability/viability/ductility probabilities, `is_metal` status, and predicted `band_gap` for insulators.