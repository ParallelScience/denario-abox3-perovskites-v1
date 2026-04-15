

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
        