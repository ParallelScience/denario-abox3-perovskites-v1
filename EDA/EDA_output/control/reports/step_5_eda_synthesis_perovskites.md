<!-- filename: reports/step_5_eda_synthesis_perovskites.md -->
# Exploratory Data Analysis Synthesis: ABO3 Perovskite Dataset

## 1. Data Health, Integrity, and Completeness

An initial assessment of the dataset's structural integrity revealed several critical data quality artifacts that must be addressed prior to downstream modeling. The raw dataset contained duplicated elemental descriptor columns (e.g., `A_site.1`, `B_site.1`), which the engineering agent successfully identified and dropped to ensure a clean feature space. 

The completeness of the dataset varies significantly across property domains:

| Feature Group | Representative Columns | Completeness | Notes |
| :--- | :--- | :--- | :--- |
| **Structural & Identifiers** | `volume`, `density`, `spacegroup_symbol` | 100% (1283/1283) | No missing values; `nelements` strictly verified as 3. |
| **Electronic & Magnetic** | `band_gap`, `is_metal`, `magnetic_ordering` | 100% (1283/1283) | No missing values. |
| **Thermodynamic** | `formation_energy_per_atom`, `energy_above_hull` | 100% (1283/1283) | `equilibrium_reaction_energy_per_atom` is only populated for the 168 stable materials (13.1%). |
| **Elastic (VRH)** | `K_VRH`, `G_VRH`, `elastic_anisotropy` | 16.8% (215/1283) | Highly sparse; contains severe unphysical outliers. |
| **Surface Properties** | `surface_energy`, `work_function` | 0% (0/1283) | Completely absent from the dataset. |
| **Elemental Descriptors** | `A_radius`, `tau`, `mu`, `en_diff` | 100% (1283/1283) | `A_period` and `B_period` are completely null. |

**Integrity Note:** There are no infinite values or duplicate `material_id` entries. However, the 100% missing columns (`surface_energy`, `work_function`, `A_period`, `B_period`) provide zero variance and must be excluded from all subsequent analyses.

## 2. Quantitative Synthesis and Property Distributions

### Electronic and Magnetic Landscape
The dataset exhibits a near-even split between metallic and non-metallic behavior, with 48.6% of the compounds classified as metals (`band_gap` = 0.0 eV). For the semiconducting/insulating fraction, the band gap distribution is broad, reaching a maximum of 6.09 eV, with a median of 0.06 eV across the entire dataset. Magnetically, 50.9% of the materials are spin-polarized. The dominant magnetic ground states are Non-Magnetic (49.1%) and Ferromagnetic (43.6%), while Ferrimagnetic (4.4%) and Antiferromagnetic (3.0%) orderings are rare.

### Structural and Thermodynamic Properties
Crystallographically, the dataset is dominated by high-symmetry systems: Cubic (33.4%) and Orthorhombic (29.9%) structures make up the majority, predominantly mapping to the `Pm-3m` (32.0%) and `Pnma` (20.3%) space groups. 

Thermodynamically, only 13.1% of the materials reside on the convex hull (`is_stable` = True). The `energy_above_hull` distribution is heavily right-skewed. While the 75th percentile is a reasonable 0.24 eV/atom, the distribution exhibits a long tail extending up to 4.16 eV/atom, with 125 statistical outliers identified via the IQR method. 

### Tolerance Factor (τ) and Structural Descriptors
The Goldschmidt tolerance factor (τ) has a mean of 0.856 and an interquartile range of [0.780, 0.921], aligning well with the theoretical stability window for perovskites. Correlation analysis reveals that τ is almost perfectly collinear with the `radius_ratio` (rA / rB, Pearson r = 0.998) and highly correlated with `A_radius` (r = 0.886). The octahedral factor (μ) shows a high outlier frequency (17.2% of the data), suggesting significant distortions in the B-O coordination polyhedra for a subset of the compounds.

## 3. Anomalies and Outlier Detection

The most critical finding of the EDA is the presence of extreme, unphysical outliers within the sparse elastic property subset (215 entries). 
- **Bulk Modulus (`K_VRH`)**: The compound `BaRuO3` (mp-3597) reports a `K_VRH` of ~1.44 × 10⁶ GPa, which is physically impossible (orders of magnitude stiffer than diamond). 
- **Shear Modulus (`G_VRH`)**: `BaRuO3` similarly reports a `G_VRH` of ~1.91 × 10⁵ GPa. Furthermore, several materials exhibit highly negative shear moduli, including `ZrOsO3` (-206.3 GPa), `YOsO3` (-105.2 GPa), `BiRhO3` (-79.5 GPa), and `NaNbO3` (-58.6 GPa). While slightly negative elastic constants can indicate dynamical instability (Born criteria violation) in DFT relaxations, these extreme magnitudes suggest severe convergence failures or pathological potential energy surfaces.

These outliers artificially dominate the variance of the elastic features and spuriously inflate the Pearson correlation between `K_VRH` and `G_VRH` to 0.999995. 

Additionally, the `volume` feature contains 88 upper-bound outliers, with a maximum of 4158 Å³ compared to a median of 179 Å³, indicating the presence of massive supercells or highly porous, uncollapsed structures.

## 4. Modeling Implications and Recommendations

Based on the statistical landscape, the following data treatments are strictly recommended for downstream machine learning and statistical modeling:

1. **Feature Pruning**: Immediately drop `surface_energy`, `work_function`, `A_period`, and `B_period`. Ensure any residual `.1` duplicate columns are removed.
2. **Elastic Outlier Clipping**: For any mechanical property prediction tasks, the extreme outliers in `K_VRH` (> 300 GPa) and `G_VRH` (> 200 GPa or < -20 GPa) must be explicitly filtered out. Failure to do so will catastrophically bias regression loss functions (e.g., MSE).
3. **Log-Transformations**: Due to orders-of-magnitude variance and heavy right-skewness, `volume` and `elastic_anisotropy` require log-transformation (log(1 + x)) prior to being used as features or targets in linear models or distance-based algorithms.
4. **Thermodynamic Target Engineering**: The heavy tail in `energy_above_hull` (> 0.58 eV/atom) likely contains unphysical or highly unrelaxed structures. If predicting stability, consider framing it as a classification task (`is_stable`), or apply a clipping threshold (e.g., 95th percentile) to the continuous target to prevent the model from over-optimizing for extreme metastable states.
5. **Optoelectronic Modeling Strategy**: Because exactly 48.6% of the dataset has a `band_gap` of 0.0 eV, standard regression models will struggle. A two-stage "hurdle" model is recommended: first, a classifier to predict metallicity (`is_metal`), followed by a regressor trained exclusively on the non-zero subset to predict the continuous band gap magnitude.
6. **Multicollinearity Mitigation**: Given the near-perfect correlation between τ and `radius_ratio`, models sensitive to multicollinearity (e.g., unregularized linear regression) should drop one of these features to maintain coefficient stability.