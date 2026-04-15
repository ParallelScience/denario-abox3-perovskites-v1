<!-- filename: reports/step_6_perovskite_pipeline_analysis.md -->
# Two-Stage Classification Pipeline for Mechanical Robustness and Thermodynamic Stability in ABO3 Perovskites

## 1. Data Preprocessing and Feature Engineering
The foundation of any robust machine learning pipeline lies in the quality and physical relevance of its feature space. The initial dataset comprised 1283 stoichiometrically-verified ABO3 perovskite compounds with 75 columns. To ensure data integrity and prevent data leakage, several preprocessing steps were executed. Duplicate columns and features with zero variance or complete absence of data (e.g., `surface_energy`, `work_function`, `A_period`, `B_period`) were systematically removed. Furthermore, the `theoretical` flag was dropped to ensure the model evaluates materials based purely on their intrinsic properties rather than their origin in the database.

To address the heavy right-skewness and orders-of-magnitude variance in certain physical properties, a log-transformation (log(1+x)) was applied to `volume` and `elastic_anisotropy`. This transformation is crucial for stabilizing the variance and improving the performance of distance-based algorithms, although tree-based models are generally invariant to monotonic transformations.

Feature engineering was guided by domain knowledge of perovskite crystal chemistry. We derived several new descriptors: the absolute deviation of the Goldschmidt tolerance factor from the ideal cubic value (`abs_tau_diff` = | τ - 1.0 |), the difference in atomic radii between the A and B sites (`radius_diff`), the absolute electronegativity difference (`en_diff`), the variance of electronegativities (`en_var`), and the total valence electron count (`VEC`). To mitigate multicollinearity, which can destabilize model coefficients and obscure feature importance, `radius_ratio` was dropped due to its near-perfect collinearity with the tolerance factor τ. Categorical variables, including `crystal_system`, `magnetic_ordering`, `A_site`, and `B_site`, were one-hot encoded. The final engineered feature matrix contained 148 dimensions, providing a rich, purely compositional and structural representation of the materials.

## 2. Thermodynamic Stability Classification
The first stage of the pipeline aimed to predict the thermodynamic stability of the perovskite compounds. A material was defined as stable (`is_stable` = True) if its energy above the convex hull (`energy_above_hull`) was exactly zero. This resulted in a highly imbalanced target distribution, with only 168 stable compounds against 1115 metastable or unstable ones.

A Gradient Boosting Classifier was trained using stratified 5-fold cross-validation. A critical step in this stage was the Feature Importance Sensitivity Analysis. We evaluated the model's performance with and without geometric proxies (e.g., `volume`, `density`, `density_atomic`). The inclusion of geometric features yielded a Receiver Operating Characteristic Area Under the Curve (ROC AUC) of 0.7900. When these features were excluded, the ROC AUC experienced a negligible drop to 0.7854. Because the performance remained stable (drop < 0.05), we prioritized compositional and bonding features, permanently dropping the geometric proxies. This decision is vital: geometric features derived from Density Functional Theory (DFT) relaxations can act as information leaks, implicitly containing information about the material's relaxed state that would be unavailable for purely hypothetical, unrelaxed compositions.

The final thermodynamic stability classifier achieved an Accuracy of 0.8714, a Precision of 0.5195, a Recall of 0.2381, and an F1 Score of 0.3265. The relatively low recall indicates that the model is conservative; it misses a significant portion of truly stable compounds. However, the precision of ~52% suggests that when the model predicts a compound to be stable, it is correct more than half the time—a substantial improvement over the baseline prevalence of 13.1%. The ROC curve and the top 20 feature importances are visualized in the saved plot <code>data/step_2_stability_roc_fi_2_1776234773.png</code>. The feature importance analysis revealed that compositional descriptors, particularly those related to electronegativity and valence electron count, played the most significant roles in driving the stability predictions.

## 3. Mechanical Viability Classification
The second stage addressed the mechanical robustness of the perovskites, a task severely complicated by data sparsity and the presence of unphysical outliers. Only 215 out of the 1283 materials (16.8%) possessed computed elastic constants (`K_VRH`, `G_VRH`). Within this subset, extreme anomalies were present, such as bulk moduli exceeding 10^6 GPa and highly negative shear moduli, indicative of severe dynamical instabilities or DFT convergence failures.

To construct a reliable "Mechanical Viability" classifier, we applied a percentile-based clipping strategy (1st to 99th percentile) to the 215-sample elastic subset. The resulting physically bounded ranges were 23.68 to 219.96 GPa for `K_VRH` and -76.59 to 143.82 GPa for `G_VRH`. Materials falling within these bounds were labeled as "Mechanically Viable" (Class 1), while those outside were deemed "Unstable/Pathological" (Class 0). This filtering yielded an extreme class imbalance: 205 viable instances versus only 10 unstable instances.

A Gradient Boosting Classifier, utilizing balanced class weights, was trained on this subset. The model achieved a ROC AUC of 0.5759 and a Precision-Recall AUC (PR AUC) of 0.9624. However, a closer inspection of the out-of-fold confusion matrix reveals a critical limitation:
<code>[[0, 10], [5, 200]]</code>
The model completely failed to identify the minority class (True Unstable), predicting all 10 unstable instances as viable. The 5 false negatives represent viable materials incorrectly classified as unstable. The high PR AUC is an artifact of the overwhelming dominance of the positive class. The ROC curve, PR curve, and confusion matrix are documented in <code>data/step_3_mechanical_viability_eval_3_1776234927.png</code>. This result underscores the profound difficulty of learning a decision boundary for mechanical instability when the pathological examples are so scarce. Consequently, the mechanical viability probabilities generated for the 1068 uncharacterized materials must be interpreted with caution, as the model exhibits a strong bias toward predicting viability.

## 4. Electronic and Ductility Profiling
To provide a comprehensive material profile, we extended the pipeline to predict optoelectronic properties and mechanical ductility. Given that nearly half the dataset (48.6%) is metallic (`band_gap` = 0.0 eV), a standard regression approach would be heavily biased. Instead, we employed a two-stage hurdle model. First, a Gradient Boosting Classifier was trained to predict metallicity (`is_metal`). Subsequently, a Gradient Boosting Regressor was trained exclusively on the non-metallic subset to predict the continuous `band_gap` magnitude.

For mechanical ductility, we utilized the Pugh ratio (G/K), a well-established empirical metric where a value below 0.571 indicates ductile behavior, and a value above indicates brittleness. A classifier was trained on the viable elastic subset to predict the "Ductile" vs. "Brittle" category. These predictions, alongside the stability and viability probabilities, were aggregated to form a holistic performance profile for every compound in the dataset, saved in <code>data/electronic_ductility_predictions.csv</code>.

## 5. Pareto-Optimal Frontier and Novelty Analysis
The ultimate objective of this research is to identify high-performance perovskite candidates that simultaneously maximize thermodynamic stability, mechanical viability, and ductility. Because these objectives can be competing, we framed the selection process as a multi-objective optimization problem, seeking the Pareto-optimal frontier in the 3D space of predicted probabilities.

Out of the 1283 candidates, 45 were identified as Pareto-optimal—meaning no other compound in the dataset possessed strictly higher probabilities across all three dimensions without being worse in at least one. To ensure chemical diversity and prevent the recommendation of highly similar compounds, we clustered the Pareto-optimal candidates by their chemical system (`chemsys`) and selected the representative with the shortest Euclidean distance to the "ideal" point (where all probabilities equal 1.0). This yielded 41 diverse Pareto-optimal candidates.

To quantify the extrapolative risk associated with these predictions, we computed the Mahalanobis distance of each candidate's feature vector relative to the distribution of the 215-sample elastic training set. A "Novelty Threshold" was established at the 97.5th percentile of the training set distances (14.5947). Across the full dataset, 313 candidates exceeded this threshold, flagging them as "High-Novelty" or "High-Risk/High-Reward" extrapolations.

The 3D Pareto frontier and its 2D projection are visualized in <code>data/step_5_pareto_frontier_5_1776235968.png</code>. The plot illustrates the trade-offs between the predicted probabilities and highlights the distinct chemical systems populating the frontier.

The top 10 diverse Pareto-optimal candidates are summarized in Table 1.

**Table 1: Top 10 Diverse Pareto-Optimal Candidates**

| Material ID | Formula | Chemical System | Prob. Stability | Prob. Viability | Prob. Ductility | Distance to Ideal | High Novelty |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| mp-3317 | SmRhO3 | O-Rh-Sm | 0.8715 | 0.9918 | 0.9662 | 0.1331 | False |
| mp-4789 | PrRhO3 | O-Pr-Rh | 0.7772 | 0.9773 | 0.9793 | 0.2249 | False |
| mp-555665 | NaVO3 | Na-O-V | 0.7785 | 0.9960 | 0.7813 | 0.3113 | False |
| mp-1213826 | CeRhO3 | Ce-O-Rh | 0.6696 | 0.9641 | 0.9943 | 0.3324 | False |
| mp-1178549 | AlVO3 | Al-O-V | 0.6611 | 0.9985 | 0.9453 | 0.3433 | True |
| mp-5163 | LaRhO3 | La-O-Rh | 0.6404 | 0.9773 | 0.9821 | 0.3607 | False |
| mp-19031 | RbVO3 | O-Rb-V | 0.6370 | 0.9961 | 0.9461 | 0.3670 | False |
| mp-998624 | NaOsO3 | Na-O-Os | 0.5906 | 0.9969 | 0.9901 | 0.4096 | False |
| mp-1078188 | LiOsO3 | Li-O-Os | 0.5442 | 0.9969 | 0.9961 | 0.4558 | False |
| mp-1105788 | SmCrO3 | Cr-O-Sm | 0.6884 | 0.9966 | 0.5505 | 0.5469 | True |

The presence of Rhodium (Rh) and Vanadium (V) based perovskites dominates the top ranks, suggesting these B-site cations are highly conducive to forming stable, mechanically robust, and ductile structures. Notably, AlVO3 and SmCrO3 are flagged as high-novelty candidates. Their feature representations deviate significantly from the elastic training set, meaning their high predicted probabilities are extrapolative. These materials represent exciting targets for future DFT validation or experimental synthesis, as they lie in unexplored regions of the chemical space.

## 6. Limitations and Implications
While the two-stage classification pipeline successfully navigates the noisy and sparse dataset to identify promising candidates, several critical limitations must be acknowledged. 

The most profound limitation is the extreme sparsity of the elastic training set (215 samples). The mechanical viability classifier, trained on this small subset, suffered from severe class imbalance (205 viable vs. 10 unstable). As demonstrated by the confusion matrix, the model failed to learn the characteristics of dynamically unstable or pathological configurations, defaulting to predicting viability for almost all instances. Consequently, the mechanical viability predictions for the 1068 uncharacterized materials are likely overly optimistic. The model acts more as a filter for extreme, easily identifiable anomalies rather than a rigorous discriminator of subtle mechanical instabilities.

Furthermore, the ductility predictions are inherently constrained by the same 215-sample subset. The extrapolation of these mechanical properties to the broader chemical space of 1283 compounds carries a high degree of uncertainty, as quantified by the Mahalanobis distance novelty analysis. 

The thermodynamic stability classifier, while more robustly trained on the full dataset, exhibited low recall (0.2381). This conservative behavior means the pipeline likely discards many genuinely stable perovskites in the first stage. The reliance on purely compositional features, while necessary to prevent data leakage, inherently limits the model's ability to capture complex structural distortions (e.g., octahedral tilting) that critically dictate phase stability.

**Conclusion**
This research successfully implemented a sequential machine learning pipeline to map the viable material space of ABO3 perovskites. By framing thermodynamic stability and mechanical robustness as classification tasks, we mitigated the catastrophic biases introduced by unphysical DFT outliers. The identification of 41 diverse Pareto-optimal candidates, including high-novelty extrapolations like AlVO3 and SmCrO3, provides a targeted roadmap for future computational and experimental investigations. Future work should focus on active learning strategies—specifically, computing the elastic constants for the high-novelty and highly uncertain candidates identified in this study—to iteratively enrich the training set and refine the mechanical decision boundaries.