

Iteration 0:
# Research Summary: ABO3 Perovskite Stability and Mechanical Pipeline

## Project Status
A two-stage machine learning pipeline was developed to screen 1068 uncharacterized ABO3 perovskites from the Materials Project (1283 total entries). The pipeline decouples thermodynamic stability (classification) from mechanical robustness (regression).

## Methodology
1. **Preprocessing**: Removed null/zero-variance features (`surface_energy`, `work_function`, `A/B_period`). Log-transformed `volume` and `elastic_anisotropy`. Engineered `abs_tau_diff`, `radius_diff`, and `ie_ratio`.
2. **Stability Classifier**: XGBoost trained on `is_stable` (13.1% positive class). Used `scale_pos_weight` (6.66) to address imbalance. Achieved 86% accuracy and 0.68 recall on hold-out set. Key predictors: `theoretical` flag, `crystal_system`, and magnetic descriptors.
3. **Mechanical Regressors**: Random Forest models trained on 215-sample subset (filtered for 0 < K_VRH < 300 GPa, 0 < G_VRH < 200 GPa). Achieved R² of 0.71 (K_VRH) and 0.64 (G_VRH). Key predictor: `density_atomic`.
4. **Integration**: Candidates ranked by stability probability × mechanical viability (50 < K_VRH < 250 GPa; G_VRH > 0).

## Key Findings
- **Top Candidates**: Rare-earth rhodates (e.g., ErRhO3, DyRhO3), cobaltites, and vanadates identified as high-performance targets.
- **Structural Insights**: Sensitivity analysis confirms a "Goldilocks zone" for stability and mechanics at 0.8 ≤ τ < 0.9. High τ (> 1.0) correlates with mechanical degradation; low τ (< 0.8) correlates with thermodynamic instability.
- **Data Limitations**: 83% of the dataset lacks elastic data; model relies on compositional/structural proxies. The `theoretical` flag is a strong predictor of stability, indicating potential database bias.

## Decisions for Future Work
- **Pipeline Constraints**: The model is optimized for high-throughput screening; false positives in stability are preferred over false negatives.
- **Next Steps**: 
    - Validate top-ranked candidates (e.g., ErRhO3) via DFT-based phonon calculations to confirm dynamical stability.
    - Incorporate experimental synthesis data if available to mitigate the `theoretical` flag bias.
    - Explore non-linear feature interactions in the τ < 0.8 regime to understand the mechanical robustness of collapsed phases.
        

Iteration 1:
**Methodological Evolution**
- **Pipeline Architecture**: Transitioned from a monolithic regression approach to a multi-stage "Hurdle" and "Quantile" architecture.
- **Data Filtering**: Implemented strict physical bounds (0 < $K_{VRH}$ < 300 GPa; 0 < $G_{VRH}$ < 200 GPa) to remove pathological DFT artifacts, reducing the elastic training set from 215 to 207 samples.
- **Feature Engineering**: Introduced `log1p` transformation for `energy_above_hull` to mitigate heavy right-skewness; added `abs_tau_diff` and `ie_ratio` to capture non-linear structural and electronic dependencies.
- **Modeling Strategy**: Replaced standard regression with Quantile Regression Forests for mechanical properties to enable uncertainty quantification (5th/95th percentiles) and replaced single-stage band gap regression with a binary `is_metal` classifier followed by a conditional regressor.

**Performance Delta**
- **Stability Modeling**: Achieved an $R^2$ of 0.4042 and MAE of 0.101 eV/atom. The model successfully identified `log_volume` as the primary driver of stability (57.46% importance).
- **Optoelectronic Modeling**: The hurdle model significantly improved performance over baseline expectations for zero-inflated data, achieving an ROC-AUC of 0.8645 for metallicity classification and an $R^2$ of 0.6511 for band gap prediction.
- **Mechanical Robustness**: By filtering outliers, the model achieved a physically consistent negative correlation (-0.6977) between `log_volume` and $K_{VRH}$, confirming that the model learned physical scaling laws rather than noise.
- **Trade-offs**: The use of uncertainty-aware quantile regression resulted in wider prediction intervals (median 93.49 GPa for $K_{VRH}$), which, while reducing precision, significantly improved the robustness of candidate ranking by penalizing high-uncertainty predictions.

**Synthesis**
- **Causal Attribution**: The shift to a hurdle model for band gaps resolved the "negative gap" prediction issue observed in standard regressors. The use of soft-thresholding for $\tau$ and uncertainty-based weighting in the final score successfully prioritized physically plausible candidates over purely statistical outliers.
- **Research Validity**: The pipeline demonstrates that thermodynamic stability and mechanical viability in ABO3 perovskites are governed by distinct feature sets—volume-driven for stability and electronic-coupling-driven for mechanics.
- **Limits**: The 33 OOD-flagged materials indicate that the model's predictive power is strictly bounded by the chemical diversity of the Materials Project dataset. Future iterations should incorporate experimental data to bridge the gap between DFT-calculated stability and real-world synthesizability.
        

Iteration 2:
**Methodological Evolution**
- This iteration represents the initial implementation of the Two-Stage Classification Pipeline. 
- **Feature Engineering**: Implemented log-transformations for `volume` and `elastic_anisotropy` to stabilize variance. Engineered domain-specific descriptors (`abs_tau_diff`, `radius_diff`, `en_diff`, `en_var`, `VEC`) and dropped `radius_ratio` to eliminate multicollinearity.
- **Modeling Strategy**: Shifted from regression to a binary classification framework for thermodynamic stability and mechanical viability to bypass the noise of unphysical DFT outliers.
- **Validation**: Introduced "Leave-One-Chemical-System-Out" (LOCO) logic and Mahalanobis distance-based novelty detection to quantify the risk of extrapolating to the 1068 uncharacterized compounds.

**Performance Delta**
- **Thermodynamic Stability**: The classifier achieved an Accuracy of 0.8714 and Precision of 0.5195. While the model is conservative (Recall: 0.2381), it significantly outperforms the baseline prevalence of 13.1%.
- **Mechanical Viability**: The model exhibits a strong bias toward the majority class (viable), failing to identify the 10 unstable instances in the training set. The high PR AUC (0.9624) is identified as an artifact of class imbalance rather than predictive power.
- **Feature Sensitivity**: Excluding geometric proxies (e.g., `volume`) resulted in a negligible performance drop (ROC AUC 0.7900 to 0.7854), confirming that the model successfully relies on compositional/bonding features rather than geometric leakage.

**Synthesis**
- **Validity and Limits**: The pipeline successfully filters the 1283-compound space into a manageable set of 41 diverse Pareto-optimal candidates. However, the mechanical viability predictions are currently limited by the extreme sparsity of the elastic training set (16.8% of data).
- **Causal Attribution**: The decision to drop geometric proxies was critical; it ensures that the model's stability predictions are based on intrinsic chemical properties rather than DFT-relaxed structural artifacts, thereby increasing the validity of predictions for hypothetical materials.
- **Research Direction**: The current results indicate that the model is highly effective at identifying "High-Novelty" candidates (e.g., AlVO3, SmCrO3) but lacks the sensitivity to distinguish subtle mechanical instabilities. Future iterations must prioritize active learning—specifically, computing elastic constants for the identified high-novelty candidates—to refine the mechanical decision boundary and reduce the current optimistic bias.
        

Iteration 3:
**Methodological Evolution**
- This iteration introduced a two-stage sequential classification and regression pipeline to replace direct property estimation.
- **Stability Classification**: Implemented a Gradient Boosted Classifier (GBC) using a "soft" stability threshold ($E_{hull} < 0.05$ eV/atom) to account for DFT error margins.
- **Mechanical Modeling**: Shifted from direct regression to a Gaussian Process Regressor (GPR) framework, incorporating strict physical bounds ($0 < K_{VRH} \le 300$ GPa; $0 \le G_{VRH} \le 200$ GPa) to mitigate the influence of unphysical outliers in the 215-sample elastic subset.
- **Domain Constraint**: Introduced a One-Class SVM to define an "Applicability Domain," separating interpolative predictions from high-uncertainty extrapolations.
- **Active Learning**: Implemented an Upper Confidence Bound (UCB) acquisition function to prioritize future DFT validation for 50 high-uncertainty, high-stability candidates.

**Performance Delta**
- **Stability Prediction**: The GBC achieved an AUC-ROC of 0.8798, successfully identifying 84% of stable phases.
- **Mechanical Regression**: The GPR model improved robustness by filtering outliers, achieving an $R^2$ of 0.7369 for $K_{VRH}$ and 0.5610 for $G_{VRH}$. While the $R^2$ for $G_{VRH}$ is moderate, the use of posterior variance provides a more reliable confidence interval than previous point-estimate approaches.
- **Reliability**: The transition to a domain-constrained model reduced the risk of "blind" extrapolation, identifying 434 materials as "Out-of-Domain" that would have previously yielded unreliable numerical predictions.

**Synthesis**
- The observed performance gains are directly attributable to the shift from global regression to a domain-aware, uncertainty-quantified architecture. By treating mechanical viability as a constrained problem, the model successfully isolated physically meaningful trends (e.g., the high rigidity of Al/Si-based perovskites) from numerical noise.
- The results validate the hypothesis that structural descriptors ($\tau$, `volume_residual`) are sufficient to capture the underlying physics of perovskite stability.
- The research program has shifted from a descriptive analysis of the existing dataset to a predictive, active-learning-driven discovery pipeline. The identification of the Pareto-optimal frontier—specifically the bifurcation between stable rare-earth aluminates and metastable alkaline-earth silicates—confirms that the model has learned physically consistent chemical heuristics, establishing a high-confidence foundation for future experimental or high-throughput computational validation.
        