# RQ3: Dimensionality Reduction & Separability Analysis

**Project:** ABIDE I — Cortical Thickness Analysis (PoliMi, Applied Statistics)  
**Focus:** Investigating cortical thickness differences between Autism Spectrum Disorder (ASD) and Control groups.

---

## 1. Objective and Rationale
The primary goal of this phase is to answer two key questions:
1. Do we need all 68 cortical thickness features, or can we reduce the dimensionality?
2. Does PCA-based dimensionality reduction make the distinction between ASD and Control groups clearer?

Because the 68 Desikan-Killiany regions are anatomically linked and not independent, the dataset contains high redundancy. Dimensionality reduction aims to capture the core biological variance with fewer features, reducing noise and preventing overfitting in downstream tasks.

## 2. Methodology
The analysis was conducted through the following systematic steps:
* **Data Validation:** Loaded and verified both harmonized (site-effect removed) and raw datasets, isolating the 68 thickness columns for the left and right hemispheres.
* **Standardization:** Applied Z-score standardization (mean=0, std=1) to ensure regions with naturally higher thickness values do not dominate the principal components.
* **Principal Component Analysis (PCA):** Extracted orthogonal components from the 68 features.
* **Variance Analysis (Scree Plot):** Calculated the cumulative explained variance to determine the optimal number of components required to retain 80%, 90%, and 95% of the original information.
* **Statistical Testing & Visualization:** Utilized Multi-Dimensional Scaling (MDS) for 2D visualization and conducted Hotelling's T² test to statistically evaluate the multivariate separability between the ASD and Control groups.

## 3. Key Findings
* **High Redundancy Confirmed:** The 68 anatomical regions showed significant covariance, confirming that adjacent regions tend to thin or thicken together.
* **Substantial Dimensionality Reduction:** Only **19 Principal Components (k=19)** are needed to explain **80%** of the total variance. This reduces the feature space to just 27.9% of its original size while preserving the vast majority of the biological information.
* **Proof of Separability:** Hotelling's T² test yielded a significantly small p-value on the full dataset. This proves that a multivariate signal distinguishing ASD from the Control group exists, strictly justifying the transition to predictive modeling (RQ4).

## 4. Limitations & Caveats
* **Linearity of PCA:** PCA only captures linear relationships. Complex, non-linear interactions distinguishing ASD from Control (e.g., curved manifolds) might be missed. 
* **Subsampling for Visualization:** The MDS visualization was performed on a smaller subsample (n=600) strictly for computational speed. However, statistical tests correctly utilized the full dataset.
* **Statistical Assumptions:** Hotelling's T² assumes multivariate normality and similar covariances. While the Central Limit Theorem helps given the sample size, future work could utilize robust alternatives (like Welch-Aspin generalizations) to ensure strict validity.

---

## 5. Next Steps: Recommendations for RQ4 (Classification)

Since RQ3 successfully proved that a statistically significant multivariate signal exists between ASD and Control groups, **RQ4** should focus on building robust binary classification models.

Here is a recommended pipeline for RQ4:

### A. Feature Selection Strategy
* **Use the PCs:** Instead of training on the raw 68 features, train the classifiers using the **19 Principal Components** identified in RQ3. This will significantly reduce the risk of the "curse of dimensionality" and prevent overfitting.
* *Alternative approach:* Compare the performance of models trained on the 19 PCs versus models trained on the raw 68 features (perhaps using L1/Lasso regularization for feature selection).

### B. Model Selection
Implement and compare a diverse set of machine learning algorithms:
1.  **Logistic Regression:** A strong baseline model that is highly interpretable.
2.  **Support Vector Machines (SVM):** Excellent for high-dimensional spaces. Try both Linear and Radial Basis Function (RBF) kernels.
3.  **Random Forest / Gradient Boosting (XGBoost, LightGBM):** Tree-based models can capture non-linear relationships that PCA might have missed.

### C. Validation & Tuning
* **Cross-Validation:** Use Stratified K-Fold Cross-Validation (e.g., 5-fold or 10-fold) to ensure models generalize well to unseen data and are not dependent on a lucky train-test split.
* **Hyperparameter Tuning:** Utilize `GridSearchCV` or `RandomizedSearchCV` to find the optimal parameters for each model (e.g., C value in SVM, number of estimators in Random Forest).

### D. Evaluation Metrics
Do not rely solely on Accuracy, especially if there is any class imbalance between ASD and Control samples. Evaluate using:
* **ROC-AUC (Area Under the Curve):** The most reliable metric for evaluating binary classifier performance across different thresholds.
* **Precision, Recall, and F1-Score:** Particularly important for understanding false positives versus false negatives in a medical/diagnostic context.
* **Confusion Matrix:** To visually inspect the exact breakdown of correct and incorrect predictions.