1. **Data Cleaning and Feature Engineering**
   - Remove `surface_energy`, `work_function`, `A_period`, `B_period`, and any duplicate columns.
   - Perform log-transformation on `volume` and `elastic_anisotropy`.
   - Map `spacegroup_number` to Glazer tilt systems using `pymatgen.symmetry.analyzer`.
   - Estimate Bond Valence Sums (BVS) for A and B sites using `A_radius` and `B_radius` as proxies for bond lengths, given the absence of explicit coordinate files.

2. **Thermodynamic Stability Classification**
   - Train a Gradient Boosted Classifier to predict `is_stable` (binary: stable vs. metastable).
   - Use BVS, Glazer tilt systems, `tau`, `mu`, and electronegativity differences as primary features.
   - Employ Leave-One-Cluster-Out (LOCO) cross-validation, grouping by A or B element to ensure generalization to unseen chemical families.
   - Evaluate performance using precision-recall curves to account for class imbalance.

3. **Robust Mechanical Modeling**
   - Train a Gaussian Process Regressor (GPR) with a Huber loss function to predict `K_VRH` and `G_VRH` using the filtered 215-sample subset (0 < K < 300 GPa, G > 0).
   - Incorporate `density`, `crystal_system` (one-hot), and BVS as features.
   - Use the GPR predictive variance ($\sigma^2$) to quantify uncertainty for the 1068 uncharacterized materials.
   - Integrate LOCO cross-validation to assess model robustness across chemical systems.

4. **Hurdle Model for Electronic Properties**
   - Implement a two-part Hurdle model for `band_gap`: a Bernoulli classifier to predict `is_metal` (gap = 0), and a truncated Log-Normal regressor trained exclusively on the non-zero gap subset.
   - Explicitly note that models are trained on PBE-level data; acknowledge that high-performance candidates require follow-up calculations (e.g., HSE06 or PBE+U) for experimental validation.

5. **Mechanical Viability Classification**
   - Train a binary classifier on the 215-sample subset to distinguish between "physically consistent" (1) and "unphysical/unstable" (0) configurations based on raw features (BVS, tilt systems, etc.).
   - This classifier provides a direct assessment of mechanical viability independent of the GPR regression output.

6. **Interpretability and Feature Importance**
   - Apply SHAP (SHapley Additive exPlanations) to the trained stability and mechanical models to decompose the contribution of physical descriptors (e.g., BVS vs. `tau`).
   - Analyze SHAP values for Pareto-optimal candidates to determine if stability is driven by ionic radii, electronegativity, or specific tilting patterns.

7. **Pareto-Front Candidate Optimization**
   - Construct a multi-objective optimization space maximizing (1) Thermodynamic Stability probability and (2) Mechanical Viability classification score.
   - Apply a penalty constraint for materials with high GPR predictive uncertainty ($\sigma^2$) to ensure physical reliability.
   - Filter the final list to exclude materials already present in the original 1283-sample dataset to prioritize novel candidates.

8. **Final Reporting and Visualization**
   - Visualize the trade-off space between stability and mechanical robustness.
   - Provide a final list of high-performance, novel candidates supported by physical insights from SHAP analysis and BVS descriptors.