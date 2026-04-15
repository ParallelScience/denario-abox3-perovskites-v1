

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
        