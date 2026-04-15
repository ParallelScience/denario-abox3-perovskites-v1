1. **Data Preprocessing and Feature Engineering**:
   - Remove columns identified as null or zero-variance: `surface_energy`, `work_function`, `A_period`, and `B_period`.
   - Drop duplicate columns (e.g., `A_site.1`, `B_site.1`) and ensure `material_id` is set as the index.
   - Apply log-transformation to `volume` and `elastic_anisotropy`.
   - Engineer new physical features: `abs_tau_diff` = `abs(tau - 1.0)`, `radius_diff` = `A_radius - B_radius`, and `ie_ratio` = `A_ie1 / B_ie1`.
   - Drop `radius_ratio` to mitigate multicollinearity with `tau`.

2. **Thermodynamic Stability Dataset Preparation**:
   - Define the target variable `is_stable` (True if `energy_above_hull` == 0, False otherwise).
   - Encode categorical features (`crystal_system`, `magnetic_ordering`) using one-hot encoding.
   - Split the data into training and hold-out test sets (80/20) to ensure no data leakage.

3. **Mechanical Property Dataset Preparation**:
   - Filter the 215-sample elastic subset to remove unphysical outliers: retain only samples where 0 < `K_VRH` < 300 GPa and 0 < `G_VRH` < 200 GPa.
   - Prepare two regression targets: `K_VRH` and `G_VRH`.
   - Ensure the training set for these regressors is strictly separated from the 1068 uncharacterized materials.

4. **Thermodynamic Stability Classifier Training**:
   - Train a Gradient Boosted Classifier (e.g., XGBoost) to predict `is_stable`.
   - Address the 13.1% class imbalance by adjusting the `scale_pos_weight` parameter or using SMOTE on the training fold.
   - Perform hyperparameter optimization using stratified k-fold cross-validation.
   - Conduct feature importance analysis (e.g., SHAP or RFE) to prune noise and ensure model interpretability.

5. **Mechanical Property Regressor Training**:
   - Train two separate regressors (e.g., Random Forest or Gradient Boosting) to predict `K_VRH` and `G_VRH` using the cleaned 215-sample subset.
   - Use structural, electronic, and magnetic features as inputs, ensuring no thermodynamic stability targets are included as features to prevent data leakage.
   - Evaluate model performance using cross-validated R² and MAE.

6. **Multi-Objective Pipeline Integration**:
   - Apply both the stability classifier and the mechanical regressors to the 1068 uncharacterized materials.
   - Define a "Mechanical Viability" score: a material is considered viable if predicted 50 GPa < `K_VRH` < 250 GPa and `G_VRH` > 0 GPa.
   - Combine the stability probability and the mechanical viability status into a final multi-objective ranking score.

7. **Model Validation and Sensitivity Analysis**:
   - Assess the robustness of the pipeline using the hold-out test set.
   - Perform sensitivity analysis to determine how variations in structural descriptors (like `tau`) impact the stability probability and predicted mechanical properties.
   - Compare the predicted "high-performance" candidates against known stable/viable subsets to verify generalization.

8. **Final Candidate Mapping**:
   - Generate a ranked list of the 1068 uncharacterized materials based on the multi-objective score.
   - Export the final dataset containing original identifiers, predicted stability probabilities, predicted elastic moduli, and the final viability ranking for downstream experimental validation.