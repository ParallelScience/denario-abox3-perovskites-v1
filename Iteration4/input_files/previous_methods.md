1. **Data Preprocessing and Feature Engineering**
   - Clean the dataset by removing `surface_energy`, `work_function`, `A_period`, `B_period`, and any duplicate columns.
   - Define a "soft" stability target: `is_stable_soft` = True if `energy_above_hull` < 0.05 eV/atom, False otherwise.
   - Calculate `VEC` (Valence Electron Count) as the sum of valence electrons for A, B, and O atoms.
   - Implement residual learning for structural features: calculate `volume_residual` as the difference between the DFT-relaxed `volume` and a geometric volume estimate $V_{geom} \approx (r_A + r_O)^2 \times (r_B + r_O) \times 8$.
   - Retain `tau`, `mu`, `en_diff`, and `VEC` as primary compositional/bonding descriptors.

2. **Thermodynamic Stability Classification**
   - Train a Gradient Boosted Classifier to predict `is_stable_soft`.
   - Use class-weight balancing to account for the imbalance between stable and metastable phases.
   - Evaluate model performance using AUC-ROC and Precision-Recall curves.

3. **Mechanical Property Regression with Uncertainty Quantification**
   - Filter the 215-sample elastic subset to remove unphysical outliers (e.g., `K_VRH` > 300 GPa, `G_VRH` < 0 or > 200 GPa).
   - Train a Gaussian Process Regressor (GPR) on the cleaned elastic subset to predict `K_VRH` and `G_VRH`.
   - Utilize the GPR’s posterior variance to quantify prediction uncertainty for each material.

4. **Domain-Constrained Prediction**
   - Train a One-Class SVM on the 215-sample elastic feature space to define the "Applicability Domain."
   - Classify materials in the 1068-sample uncharacterized set as "In-Domain" or "Out-of-Domain" based on the One-Class SVM decision function.
   - Flag "Out-of-Domain" materials as "High-Uncertainty" candidates.

5. **Active Learning Candidate Selection**
   - Implement an acquisition function (e.g., Upper Confidence Bound) that balances the predicted stability probability (exploitation) with the GPR prediction uncertainty (exploration).
   - Rank the 1068 uncharacterized materials using this acquisition function.
   - Select 50 candidates that maximize the acquisition score for future DFT validation.

6. **Ductility and Electronic Profiling**
   - Train a classifier to predict `is_metal` using the full dataset.
   - For the non-metallic subset, train a Random Forest regressor to predict `band_gap`.
   - Calculate the `pugh_ratio` ($G/K$) for materials within the Applicability Domain. Use Monte Carlo sampling of the GPR posterior distributions for $K$ and $G$ to propagate uncertainty and report a confidence interval for the `pugh_ratio`.

7. **Pareto-Optimal Frontier Analysis**
   - Construct a multi-objective optimization space using: (1) Predicted Stability Probability, (2) Predicted Mechanical Viability (GPR mean), and (3) Predicted Band Gap.
   - Identify the Pareto-optimal frontier of materials that maximize stability and mechanical robustness.
   - Cluster these candidates by chemical system to ensure diversity in the final recommendations.

8. **Final Reporting and Documentation**
   - Compile the final candidate list, including `material_id`, stability probabilities, predicted mechanical properties with uncertainty bounds, and electronic status.
   - Clearly distinguish between "Validated" candidates (within the Applicability Domain) and "Active Learning" candidates (high-uncertainty, high-stability).