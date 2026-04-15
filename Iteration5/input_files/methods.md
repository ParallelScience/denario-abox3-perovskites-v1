1. **Data Preprocessing and Feature Engineering**
   - Clean the dataset by removing `surface_energy`, `work_function`, `A_period`, `B_period`, and any duplicate columns.
   - Calculate the Valence Electron Count (VEC) for each compound.
   - Generate structural proxies: use `tau` (tolerance factor) and `mu` (octahedral factor) as primary descriptors for structural distortion. If structure files are accessible via `pymatgen`, compute B-O-B bond-angle variance; otherwise, use `spacegroup_number` and `crystal_system` (one-hot encoded) as categorical proxies for symmetry-breaking.
   - Encode `magnetic_ordering` and `crystal_system` using one-hot encoding.

2. **Thermodynamic Stability Ranking (LambdaMART)**
   - Frame stability prediction as a learning-to-rank problem.
   - Train a LambdaMART model to rank materials by their proximity to the convex hull, where lower `energy_above_hull` values correspond to higher stability ranks.
   - Include `is_magnetic`, `total_magnetization`, `tau`, `mu`, and the structural distortion proxy as features.
   - Validate using Normalized Discounted Cumulative Gain (NDCG) to ensure the ranking effectively prioritizes stable phases.

3. **Uncertainty-Aware Mechanical Regression (GPR)**
   - Filter the 215-sample elastic subset to remove unphysical outliers ($K_{VRH} > 300$ GPa, $G_{VRH} < 0$ GPa, or $K_{VRH} \leq 0$ GPa).
   - Train separate Gaussian Process Regressors (GPR) for $K_{VRH}$ and $G_{VRH}$ using `density`, `crystal_system` (one-hot), and composition-based features.
   - Derive the `Pugh_ratio` ($G/K$) from the GPR predictions and propagate the predictive variances ($\sigma^2$) to quantify uncertainty for the 1068 uncharacterized materials.

4. **Electronic Hurdle Modeling**
   - Implement a two-stage hurdle model: first, a classifier to predict `is_metal` (band_gap = 0).
   - For the non-metallic subset, train a Random Forest regressor to predict the continuous `band_gap`.
   - Evaluate performance using RMSE and Mean Absolute Error, accounting for the systematic underestimation of PBE band gaps.

5. **Pareto-Front Candidate Optimization**
   - Construct a multi-objective optimization space using: (1) Stability Rank (from LambdaMART) and (2) Mechanical Uncertainty (from GPR).
   - Identify the Pareto-optimal set of candidates—those that cannot be improved in stability rank without increasing mechanical uncertainty.
   - Visualize the trade-off space, explicitly labeling axes as "Stability Rank" and "Mechanical Uncertainty" to highlight candidates that are both stable and reliably predicted.

6. **Applicability Domain and Diversity Filtering**
   - Perform PCA on the feature space (excluding target variables to prevent data leakage) to visualize the distribution of the 1068 uncharacterized materials relative to the training set.
   - Apply a diversity filter by clustering candidates based on `chemsys` to ensure the final selection represents a broad range of chemical compositions.
   - Exclude candidates predicted to be metallic to focus on high-performance insulators and semiconductors.

7. **Sensitivity Analysis**
   - Perform sensitivity analysis on Pareto-optimal candidates by perturbing input features (e.g., ionic radii) within their standard error ranges.
   - Assess the stability of the `Pugh_ratio` predictions under these perturbations to classify candidates as "robustly ductile" or "robustly brittle."
   - Quantify confidence intervals for the predicted properties of the top-ranked candidates.

8. **Final Reporting and Synthesis**
   - Compile the final list of candidates residing on the Pareto front.
   - Map the relationship between structural descriptors and predicted performance metrics.
   - Document findings, emphasizing the physical insights gained from the transition to uncertainty-aware regression and rank-based stability modeling.