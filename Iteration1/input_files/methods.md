1. **Data Cleaning and Feature Engineering**:
   - Remove columns: `surface_energy`, `work_function`, `A_period`, `B_period`, `theoretical`, and any duplicate columns. Set `material_id` as the index.
   - Apply log-transformation to `volume` and `elastic_anisotropy`.
   - Engineer features: `abs_tau_diff` = `abs(tau - 1.0)`, `radius_diff` = `A_radius - B_radius`, and `ie_ratio` = `A_ie1 / B_ie1`.
   - Drop `radius_ratio` to mitigate multicollinearity with `tau`.
   - Ensure all features used for predictive modeling are strictly compositional or structural (e.g., A_Z, B_Z, electronegativity, radii) to prevent data leakage from DFT-calculated properties.

2. **Thermodynamic Stability Modeling**:
   - Define the target as `log1p(energy_above_hull)` to handle the heavy right-skewness of the distribution.
   - Train a Gradient Boosted Regressor to predict the transformed stability target.
   - Use cross-validation to ensure the model relies on physical descriptors rather than metadata.
   - Implement a soft-thresholding penalty function (Gaussian-based) centered at 0.85 for `tau` during the final candidate ranking phase to favor physically plausible perovskite structures.

3. **Optoelectronic Hurdle Modeling**:
   - Implement a two-stage hurdle model for `band_gap`:
     - Stage 1: A classifier to predict `is_metal` (binary).
     - Stage 2: A regressor trained exclusively on the non-metallic subset to predict the continuous `band_gap` magnitude.
   - Use these predictions to characterize the electronic profile of the 1068 uncharacterized materials.

4. **Mechanical Property Prediction with Quantile Regression**:
   - Filter the 215-sample elastic subset to retain only physically valid entries: 0 < `K_VRH` < 300 GPa and 0 < `G_VRH` < 200 GPa.
   - Train Quantile Regression Forests to predict `K_VRH` and `G_VRH`, extracting the 5th and 95th percentiles to quantify prediction uncertainty.
   - Exclude candidates where the prediction interval width exceeds 20% of the median predicted value.
   - Include `is_metal` and the predicted `band_gap` as independent features to capture potential electronic-mechanical coupling.

5. **Ductility Classification**:
   - Calculate `pugh_ratio` for the 215-sample subset.
   - Train a classifier to categorize materials into "Ductile" (Pugh < 0.57) or "Brittle" (Pugh > 0.57).
   - Apply this classifier to the 1068 uncharacterized materials to provide mechanical manufacturing insights.

6. **Multi-Objective Pipeline Integration and Filtering**:
   - Apply all trained models to the 1068 uncharacterized materials.
   - Define a "High-Performance Score" as a weighted combination of: (a) low predicted `energy_above_hull`, (b) high mechanical confidence (low uncertainty), and (c) desired ductility class.
   - Apply the soft-thresholding `tau` filter to the final ranking.

7. **Validation and Physical Consistency Checks**:
   - Assess the pipeline using a 20% hold-out test set.
   - Perform Out-of-Distribution (OOD) detection using Mahalanobis distance in the feature space to flag materials significantly different from the training set.
   - Use Partial Dependence Plots (PDPs) to verify that the model captures non-linear physical trends (e.g., stability vs. `tau`).
   - Verify physical consistency: ensure the model predicts higher bulk moduli for materials with smaller unit cell volumes.

8. **Final Candidate Mapping**:
   - Generate a ranked list of the 1068 uncharacterized materials based on the multi-objective score.
   - Export the final dataset containing `material_id`, predicted `energy_above_hull`, predicted `K_VRH` and `G_VRH` with 90% confidence intervals, ductility classification, and the final viability ranking.