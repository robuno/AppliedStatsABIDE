# The Quest for Autism Subtypes: A Deep Dive into Cortical Thickness Clusters

## Introduction: The Scientific Premise
Our scientific journey in Research Question 2 (RQ2) started with a fundamental hypothesis: **If Autism Spectrum Disorder (ASD) is truly a spectrum, can we identify distinct, neurobiologically coherent subgroups purely based on brain morphology?** 

We hypothesized that unsupervised machine learning could find hidden patterns in the cortical thickness of 68 brain regions, separating ASD patients into clinically meaningful subtypes that perhaps align with their symptom severity (measured by ADOS scores). What follows is the story of this experiment—our initial attempts, the confounders we encountered, the mathematical methods we deployed to counter them, and our final revelations.

---

## Act I: The Unsupervised Subtyping Attempt
**Notebook:** `RQ2_Unsupervised_Subtyping.ipynb`

Our first step was to throw the raw (but site-harmonized) data at the wall to see what stuck. We used four different clustering algorithms:
- **K-Means & K-Medoids:** Centroid-based algorithms assuming spherical clusters.
- **Agglomerative Hierarchical:** Building a tree (dendrogram) of similarities.
- **DBSCAN:** A density-based algorithm looking for continuous dense regions.

To find the optimal number of clusters ($k$), we relied on the **Elbow Method** (minimizing within-cluster sum of squares) and the **Silhouette Score** (measuring how similar a point is to its own cluster compared to others).

**What we found:** The algorithms successfully found mathematical clusters. However, when we checked if these clusters aligned with clinical severity (ADOS scores), the answer was a resounding *no*. The clusters were biologically real, but they weren't separating "severe ASD" from "mild ASD". They were separating something else.

---

## Act II: The Invisible Confounder - Age Residualization
**Notebook:** `RQ2_Direction1_AgeResidual.ipynb`

We quickly discovered the culprit driving the clusters: **Age**. The cortex naturally thins as humans grow older. Our unsupervised algorithms weren't finding ASD subtypes; they were simply clustering "older brains" and "younger brains."

To defeat this confounder, we employed **Age Residualization**. 

### The Mathematics of Residualization
We needed to subtract the normal aging process from the data. Crucially, to avoid circular reasoning, we fit a linear model *only on the neurotypical Control subjects*:
$$ \hat{Thickness}_{Control} = \beta_0 + \beta_1 \times Age $$

This equation tells us what a typical brain's thickness should be at any given age. We then applied this exact model to the ASD subjects and calculated the **residual** (the deviation from the typical trajectory):
$$ Residual_{ASD} = Thickness_{Observed} - \hat{Thickness}_{Predicted(from\ Controls)} $$

**What we found:** The age confounder was successfully destroyed (age correlation reduced by 76%). We clustered the residuals, forcing the algorithm to find structures *orthogonal* to age. 
**What we couldn't find:** Even with age removed, the clusters *still* did not align with ADOS scores ($\eta^2 = -0.003$). The 68-region space was simply too noisy.

---

## Act III: Filtering the Noise and Finding Asymmetry
**Notebook:** `RQ2_Direction3_Features.ipynb`

Realizing that 68 regions contained too much irrelevant data, we pivoted to severe feature engineering. 

1. **Strict Filtering:** We threw away 60 regions and kept only the 8 regions that were definitively proven to be affected by ASD in our earlier RQ1 analysis. 
2. **The Asymmetry Index (AI):** Cortical asymmetry (left vs right hemisphere differences) is largely age-independent and heavily implicated in neurodevelopmental disorders.

### The Mathematics of Asymmetry
For each region pair, we calculated:
$$ AI = \frac{Thickness_{Left} - Thickness_{Right}}{Thickness_{Left} + Thickness_{Right}} $$
- $AI > 0$ means leftward asymmetry.
- $AI < 0$ means rightward asymmetry.

**What we found:** By clustering on this highly focused, noise-free, age-independent feature space, we obtained far more stable and biologically localized clusters. Yet, the direct mapping to a single clinical severity number (ADOS) remained elusive, suggesting ASD severity is highly highly multi-dimensional.

---

## Act IV: Deep Analysis - Global vs. Regional Patterns
**Notebooks:** `RQ2_DeepAnalysis.ipynb` & `RQ2_DeepAnalysis_EN.ipynb`

To truly understand what our clusters were separating, we cracked open the "black box" using **Principal Component Analysis (PCA)**. 

When PCA reduces our data into components, we can look at the **loadings** (weights) of each brain region on the first Principal Component (PC1)—the main axis of separation.

### Hypothesis Testing via PCA Loadings
- **H1 (Global Thickness):** If all regions have positive loadings on PC1, the algorithm is just separating "overall thick brains" from "overall thin brains."
- **H2 (Regional Subtype):** If some regions have positive loadings and others negative, PC1 represents a localized *contrast*—a true neuro-anatomical subtype.

**What we found:** The analysis revealed that clustering on 68 regions often defaults to a Global Thickness separation (H1). However, when we restricted the clustering to the 8 proven RQ1 regions or used specific residualized data, we began to uncover true Regional Subtypes (H2).

---

## Act V: The Supervised Pivot
**Notebook:** `RQ2_AdosClusters.ipynb`

Having exhausted unsupervised methods, we asked a direct question: Can we explicitly find the mathematical axes that separate clinical severity?

We utilized **Linear Discriminant Analysis (LDA)** and **Random Forest Classifiers**.
While PCA maximizes *variance* regardless of labels, LDA maximizes the *separation between known classes* (in this case, discrete ADOS severity levels 1-10).

**What we found:** By forcing the algorithms to look at ADOS scores (supervised learning), we could identify specific regional profiles that correlate with severity. However, attempting to predict an exact ADOS score using a Random Forest Classifier proved incredibly difficult. 

---

## Conclusion: The Grand Summary

### What We Found
1. **Age is a massive driver of brain morphology.** Any clustering done without rigorous Age Residualization will simply rediscover the aging process.
2. **Whole-brain clustering is noisy.** Using all 68 regions tends to cluster patients based on global brain size/thickness rather than localized ASD pathologies.
3. **Targeted features work better.** Focusing only on the 8 regions identified in RQ1, and utilizing the Asymmetry Index, yields much more stable, neurobiologically sound groupings.

### What We Couldn't Find
1. **A simple map from thickness to severity.** We could not find a straightforward, unsupervised mapping where Cluster A equals "Mild ASD" and Cluster B equals "Severe ASD". 
2. **High predictive accuracy.** Cortical thickness alone, even when processed with advanced machine learning classifiers, is not sufficient to accurately predict a patient's exact ADOS score. The structural variations in autism are too heterogeneous to be reduced to a single severity axis.

Our experiment successfully mapped the landscape of ASD cortical morphology, proving that while anatomical subtypes exist, clinical severity is a deeply complex, multi-dimensional phenomenon that resists simple clustering.
