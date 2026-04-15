1. **Data Cleaning and Feature Engineering**
   - Remove `surface_energy`, `work_function`, `A_period`, `B_period`, and any duplicate columns.
   - Calculate `VEC` (Valence Electron Count) for each compound.
   - Compute `volume_residual` as the difference between the DFT-relaxed `volume` and the geometric estimate $V_{geom} \approx (r_A + r_O)^2 \times (r_B + r_O) \times 8$.
   - Generate a "tilt proxy" feature: the ratio of the actual `volume` to the ideal cubic volume calculated from the Goldschmidt tolerance factor ($\tau$) and ionic radii.
   - Ensure these geometric features are calculated consistently across the entire 1283-compound set.

2. **Thermodynamic Stability Classification**
   - Define `is_stable` based on `energy_above_hull` == 0.
   - Train a Gradient Boosted Classifier to predict `is_stable`.
   - Perform a decomposition pathway analysis on "false positives" (metastable phases predicted as stable) by comparing them against the `equilibrium_reaction_energy_per_atom` and identifying competing non-perovskite polymorphs. Label these as "structurally metastable" if they are significantly higher in energy than known competing structures.

3. **Mechanical Viability Filtering**
   - Filter the 215-sample elastic subset to retain only entries where 0 < `K_VRH` < 300 GPa and 0 < `G_VRH` < 200 GPa.
   - Train a binary classifier using stratified cross-validation to distinguish between "mechanically robust" and "dynamically unstable" configurations.
   - Include `tau`, `mu`, `volume_residual`, and "tilt proxy" as primary features.
   - Use SHAP or permutation feature importance to confirm that structural descriptors are the primary drivers of the model's decisions.

4. **Silicate Paradox and Pressure-Volume Analysis**
   - Isolate silicate-based perovskites versus non-silicate perovskites.
   - Calculate the Pearson correlation coefficient between `volume_residual` and `energy_above_hull` for both subsets to determine if negative `volume_residual` is a unique signature of silicate instability.
   - Quantify the relationship between `volume_residual` and the model's stability probability to validate the physical basis of the predictions.

5. **Electronic Property Hurdle Modeling**
   - Implement a two-stage hurdle model: first, a classifier to predict `is_metal` (band_gap = 0).
   - For the non-metallic subset, train a Random Forest regressor to predict the continuous `band_gap`.
   - Evaluate performance using RMSE and mean absolute error, accounting for the PBE functional's systematic underestimation.

6. **Applicability Domain and Boundary Analysis**
   - Train a One-Class SVM on the cleaned elastic feature space to define the "Applicability Domain."
   - Perform PCA on the feature space to visualize the distribution of the 1068 uncharacterized materials.
   - Identify "near-boundary" candidates—materials residing at the edge of the In-Domain cluster—using Euclidean distance in the PCA-transformed space.

7. **Candidate Selection and Ranking**
   - Rank the 1068 uncharacterized materials based on: (1) High stability probability, (2) Mechanical viability, and (3) Proximity to the Applicability Domain boundary.
   - Apply a diversity filter (clustering by `chemsys`) to the final selection to ensure a broad chemical representation.
   - Exclude any candidates predicted to be metallic to focus on high-performance insulators/semiconductors.

8. **Sensitivity Analysis and Reporting**
   - Calculate the `pugh_ratio` ($G/K$) for the selected candidates.
   - Perform a sensitivity analysis by varying $K$ and $G$ by their respective RMSE values to classify candidates as "ductile" or "brittle" with associated confidence intervals.
   - Compile the final report, mapping the relationship between structural descriptors (tilt, volume residual) and predicted material performance.