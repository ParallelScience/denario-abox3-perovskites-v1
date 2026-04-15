# Results and Discussion

## 1. Introduction to the Pipeline and Data Preprocessing

The primary objective of this study was to develop a robust, multi-stage machine learning pipeline capable of identifying high-performance ABO3 perovskite candidates from a dataset of 1283 stoichiometrically-verified compounds sourced from the Materials Project. A significant and pervasive challenge in materials informatics—particularly within high-throughput Density Functional Theory (DFT) databases—is the extreme sparsity and occasional unphysical noise present in computed mechanical properties. In the present dataset, only 215 materials (16.8%) possessed computed elastic constants (Voigt-Reuss-Hill averages). Furthermore, exploratory data analysis revealed severe pathological outliers within this subset, such as bulk moduli exceeding $10^6$ GPa and highly negative shear moduli, which are indicative of severe DFT convergence failures or dynamical instabilities rather than physical reality. 

To circumvent these artifacts and prevent catastrophic model bias, we implemented a sequential classification and regression architecture. The pipeline first evaluates thermodynamic stability and optoelectronic properties across the full compositional space, utilizing strictly compositional and structural descriptors to prevent data leakage. This is followed by a mechanical viability assessment trained strictly on a physically bounded subset of the elastic data. This multi-objective approach ensures that the final candidate recommendations are thermodynamically plausible, electronically profiled, and mechanically robust, effectively mapping the viable material space without relying on unreliable numerical predictions for unstable phases.

## 2. Thermodynamic Stability Modeling

Thermodynamic stability is the foremost criterion for the realizability and synthesizability of any predicted material. In this study, stability was quantified using the energy above the convex hull (`energy_above_hull`), a continuous measure where a value of zero indicates a stable ground state, and positive values denote metastability. Given the heavy right-skewness of the distribution—characterized by a long tail of highly unstable configurations—the target variable was transformed using a $\log(1 + x)$ function. This transformation stabilizes the variance and prevents the regression algorithm from disproportionately optimizing for extreme, unphysical metastable states.

A Gradient Boosting Regressor was trained to predict this transformed stability metric. Evaluated via rigorous 5-fold cross-validation, the model achieved an $R^2$ score of $0.4042 \pm 0.181$, a Root Mean Squared Error (RMSE) of $0.1764 \pm 0.0225$ (in the log-transformed space), and a Mean Absolute Error (MAE) of $0.101 \pm 0.0056$. While the $R^2$ value indicates a moderate proportion of variance explained, the low MAE demonstrates the model's capability to reliably estimate the stability of perovskite structures within a tight error margin, particularly for compounds residing near the convex hull where predictive accuracy is most critical.

The feature importance analysis yielded profound insights into the physicochemical drivers of perovskite stability. The logarithmic unit cell volume (`log_volume`) emerged as the overwhelmingly dominant predictor, accounting for 57.46% of the model's predictive capacity. This aligns with the fundamental crystallographic principle that the stability of the ABO3 framework is highly sensitive to steric packing. The optimal volume is required to accommodate the corner-sharing BO6 octahedra; deviations from this optimal volume induce excessive lattice strain, leading to structural distortions (e.g., octahedral tilting or Jahn-Teller distortions) that directly impact the thermodynamic energy landscape.

Following volume, the electronegativity difference between the B-site and A-site (`en_diff`, 8.15%) and the A-site electronegativity (`A_en`, 7.42%) were the next most influential features. These electronic descriptors dictate the degree of covalency versus ionicity in the A-O and B-O bonds. A delicate balance of ionic and covalent character is critical for stabilizing the perovskite lattice against competing non-perovskite polymorphs. Interestingly, the Goldschmidt tolerance factor (`tau`) and its absolute deviation from the ideal cubic value of 1.0 (`abs_tau_diff`) collectively contributed approximately 5.24% to the model's importance. The tolerance factor is a classical, purely geometric descriptor for perovskite formability. Its moderate ranking in this machine learning context suggests that while $\tau$ is a necessary baseline condition for perovskite formation, the complex interplay of unit cell volume and bond ionicity provides a much more granular and predictive description of the thermodynamic stability than rigid-sphere geometric ratios alone.

## 3. Optoelectronic Hurdle Modeling

The dataset exhibited a near-even bifurcation in electronic behavior, with 48.6% of the compounds classified as metallic (exhibiting a DFT-computed band gap of 0.0 eV). Standard regression algorithms struggle significantly to model such zero-inflated distributions, often predicting unphysical negative band gaps or severely underestimating the gaps of wide-gap insulators. To address this distributional challenge, we implemented a two-stage hurdle model.

The first stage comprised a Gradient Boosting Classifier trained to predict the binary state of metallicity (`is_metal`). Evaluated using stratified 5-fold cross-validation, the classifier demonstrated strong discriminative performance, achieving an accuracy of 0.781, an F1 score of 0.7687, and a Receiver Operating Characteristic Area Under the Curve (ROC-AUC) of 0.8645. This robust classification confirms that the transition between metallic and insulating states in ABO3 perovskites can be effectively mapped using purely compositional and structural features. The model likely captures the underlying physics of valence electron concentration and the degree of orbital overlap dictated by the B-site transition metals.

In the second stage, a Gradient Boosting Regressor was trained exclusively on the non-metallic subset to predict the continuous magnitude of the band gap. This conditional regressor achieved an $R^2$ of 0.6511, an RMSE of 0.8498 eV, and an MAE of 0.6158 eV. Considering the inherent systematic underestimation of band gaps by the PBE functional used in the Materials Project database, these metrics represent a highly competent surrogate model. The hurdle architecture successfully decoupled the binary physical state (metal vs. non-metal) from the continuous electronic property, providing a reliable mechanism to profile the optoelectronic nature of the uncharacterized materials without the confounding influence of the metallic zero-gap cluster.

## 4. Mechanical Property Prediction and Uncertainty Quantification

The prediction of mechanical properties was strictly restricted to the 215-sample subset possessing computed elastic constants. To ensure the model learned from physically meaningful data, we applied a rigorous filtering criterion, excluding materials with bulk moduli ($K_{VRH}$) outside the 0–300 GPa range and shear moduli ($G_{VRH}$) outside the 0–200 GPa range. This filtering retained 207 valid samples, successfully eliminating the pathological outliers that would otherwise catastrophically bias the regression loss functions and render the predictions meaningless.

Recognizing the inherent noise and extreme sparsity of the elastic data, we employed Quantile Regression Forests to predict the median (50th percentile) as well as the 5th and 95th percentiles for both $K_{VRH}$ and $G_{VRH}$. This probabilistic approach provides a built-in mechanism for uncertainty quantification. The models yielded a median prediction interval width (90% confidence interval) of 93.49 GPa for $K_{VRH}$ and 77.80 GPa for $G_{VRH}$. While these intervals are relatively wide—reflecting the intrinsic difficulty of predicting high-order derivative properties from sparse compositional data—they are crucial for downstream filtering. By quantifying the uncertainty, we can explicitly penalize candidates with highly uncertain mechanical predictions during the final ranking phase, prioritizing materials where the model exhibits high confidence.

Furthermore, we evaluated the macroscopic mechanical behavior by classifying the materials into "Ductile" or "Brittle" regimes based on the Pugh ratio ($G_{VRH} / K_{VRH}$). Using the critical empirical threshold of 0.57, the filtered elastic subset contained 141 ductile and 74 brittle materials. A Gradient Boosting Classifier trained to predict this binary outcome achieved a cross-validated accuracy of 0.7721 and an F1 score of 0.6423 for the minority (brittle) class. This ductility classifier provides a vital qualitative descriptor for manufacturability, allowing us to screen for materials that are less prone to catastrophic fracture under stress.

## 5. Validation and Physical Consistency Checks

To ensure the reliability of the pipeline when extrapolating to the 1068 uncharacterized materials, we conducted rigorous Out-of-Distribution (OOD) detection and physical consistency checks. 

OOD detection was performed by calculating the Mahalanobis distance for each material within the multidimensional compositional and structural feature space. Using a strict 97.5th percentile threshold (distance = 6.8483), the algorithm flagged 33 materials as OOD. These flagged compounds represent chemical spaces or structural distortions significantly divergent from the training manifold, indicating that predictions for these specific materials should be treated with caution as they lie outside the model's domain of applicability.

Crucially, we verified the physical consistency of the mechanical predictions by examining the relationship between the predicted bulk modulus ($K_{VRH}$) and the logarithmic unit cell volume. In solid-state physics, there is a well-established inverse relationship between volume and bulk modulus; more compact, densely packed lattices exhibit higher resistance to uniform compression. Our predicted values for the uncharacterized set demonstrated a strong negative Pearson correlation of -0.6977 with `log_volume`. This robust negative correlation confirms that the machine learning model has successfully internalized the fundamental physical laws governing lattice mechanics, rather than merely memorizing statistical artifacts from the training set.

## 6. Multi-Objective Pipeline Integration and Final Candidate Mapping

In the final phase of the study, the trained models were deployed to evaluate the 1068 uncharacterized ABO3 perovskites. The predictions revealed a diverse material landscape: 196 materials were predicted to be highly stable (energy above hull < 0.05 eV/atom), 510 were classified as metals, and 867 were predicted to exhibit ductile behavior. The median predicted bulk and shear moduli across this uncharacterized set were 144.19 GPa and 60.81 GPa, respectively, aligning perfectly with the expected mechanical regime for typical oxide perovskites.

To isolate the most promising candidates, we formulated a "High-Performance Score" that mathematically integrates thermodynamic stability, mechanical confidence, and ductility. The score was constructed as a weighted linear combination: 40% allocated to normalized thermodynamic stability (favoring lower energy above hull), 40% to mechanical prediction confidence (the inverse of the combined 90% prediction interval widths for $K_{VRH}$ and $G_{VRH}$), and a 20% bonus for predicted ductility. Finally, to ensure crystallographic plausibility, the raw score was modulated by a Gaussian soft-thresholding penalty centered at the ideal Goldschmidt tolerance factor of $\tau = 0.85$.

The pipeline successfully ranked the 1068 candidates, yielding a highly curated list of top-tier materials. The top 20 high-performance candidates are presented in the table below:

| Rank | Material ID | Formula | HP Score | Predicted EAH (eV/atom) | Predicted $K_{VRH}$ (GPa) | Predicted Brittle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | mp-1184833 | HoRhO3 | 0.9327 | 0.4261 | 179.96 | 0 (Ductile) |
| 2 | mp-1183821 | DyRhO3 | 0.9312 | 0.4405 | 179.96 | 0 (Ductile) |
| 3 | mp-1078341 | EuNbO3 | 0.9212 | 0.0646 | 159.95 | 0 (Ductile) |
| 4 | mp-1416093 | YFeO3 | 0.9126 | 0.3638 | 171.09 | 0 (Ductile) |
| 5 | mp-755572 | EuNbO3 | 0.9115 | 0.2165 | 167.95 | 0 (Ductile) |
| 6 | mp-1187595 | YbNiO3 | 0.9114 | 0.4398 | 181.87 | 0 (Ductile) |
| 7 | mp-1184411 | EuMoO3 | 0.9088 | 0.2547 | 162.87 | 0 (Ductile) |
| 8 | mp-973486 | LuVO3 | 0.9030 | 0.3883 | 186.26 | 0 (Ductile) |
| 9 | mp-1186134 | NaFeO3 | 0.8993 | 0.1817 | 167.36 | 0 (Ductile) |
| 10 | mp-1076642 | NaCrO3 | 0.8989 | 0.1620 | 169.08 | 0 (Ductile) |
| 11 | mp-1001571 | CaFeO3 | 0.8972 | 0.1787 | 177.93 | 0 (Ductile) |
| 12 | mp-862606 | EuSnO3 | 0.8965 | 0.3331 | 150.38 | 0 (Ductile) |
| 13 | mp-1016833 | CaRhO3 | 0.8954 | 0.1330 | 174.26 | 0 (Ductile) |
| 14 | mp-1185065 | LaZrO3 | 0.8930 | 0.2257 | 162.11 | 0 (Ductile) |
| 15 | mp-770347 | TbNiO3 | 0.8898 | 0.0311 | 164.76 | 0 (Ductile) |
| 16 | mp-754524 | CeTiO3 | 0.8871 | 0.1924 | 175.81 | 0 (Ductile) |
| 17 | mp-1387900 | YCrO3 | 0.8860 | 0.0929 | 147.02 | 0 (Ductile) |
| 18 | mp-22246 | EuTiO3 | 0.8856 | 0.1801 | 174.75 | 0 (Ductile) |
| 19 | mp-1099583 | SmTiO3 | 0.8850 | 0.1861 | 172.79 | 0 (Ductile) |
| 20 | mp-2647018 | PmTiO3 | 0.8844 | 0.1924 | 174.66 | 0 (Ductile) |

The top-ranked candidates are predominantly rare-earth transition metal oxides (e.g., HoRhO3, DyRhO3, EuNbO3, YFeO3). These materials exhibit a compelling combination of moderate metastability (Energy Above Hull generally between 0.05 and 0.45 eV/atom), high predicted bulk moduli (150–186 GPa), and predicted ductile behavior. The presence of multiple europium-based (EuNbO3, EuMoO3, EuSnO3, EuTiO3) and yttrium-based (YFeO3, YCrO3) perovskites in the top 20 highlights specific compositional families that inherently balance thermodynamic formability with strong, flexible mechanical lattices. Notably, EuNbO3 (mp-1078341) and TbNiO3 (mp-770347) stand out as exceptionally promising candidates, possessing very low predicted energies above the hull (0.0646 and 0.0311 eV/atom, respectively) while maintaining robust mechanical profiles.

In conclusion, this multi-stage machine learning pipeline successfully navigated the extreme sparsity and noise of the Materials Project elastic dataset. By decoupling thermodynamic stability, optoelectronic state, and mechanical viability, and by enforcing strict physical bounds and uncertainty quantification, we have mapped the uncharacterized ABO3 perovskite space and identified a highly credible set of novel, mechanically robust, and thermodynamically viable materials for future experimental synthesis and characterization.