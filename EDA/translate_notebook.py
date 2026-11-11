import json
import os

notebook_path = 'rq2/RQ2_DeepAnalysis.ipynb'
output_path = 'rq2/RQ2_DeepAnalysis_EN.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the new English sources for all 28 cells
sources = {}

sources[0] = """# RQ2 — Deep Analysis: Understanding Cluster Separation
### ABIDE I · Cortical Thickness · ASD Subtype Analysis

**Goal of this notebook:**
- Understanding what clustering separates (Global thickness? Regional subtype?)
- 68 regions vs 8 RQ1 regions comparison — which one produces a better cluster?
- PCA loading analysis — which regions drive the separation?
- Maximum discriminative axis with LDA
- Measuring the effect of harmonization and age correction
- Testing H1 (Global) vs H2 (Regional) hypotheses

**Blocks:**
1. Imports & Setup
2. Data Preparation — 3 Pipelines (Raw / ComBat / Residualized)
3. 68 Regions vs 8 Regions Clustering Comparison
4. PCA Loading Analysis — Who Drives the Separation?
5. H1 vs H2 Hypothesis Testing
6. LDA — Maximum Discriminative Axis
7. Cluster Profile Analysis — What are the Two Groups Anatomically?
8. Pipeline Effect — Contribution of Harmonization & Age Correction
9. General Summary
"""

sources[1] = """## Block 1 — Imports & Setup"""

sources[2] = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.spines.top': False,
    'axes.spines.right': False
})
SEED = 42
np.random.seed(SEED)

RQ1_REGIONS = [
    'lh_lateraloccipital', 'rh_lateraloccipital',
    'lh_lingual',          'rh_lingual',
    'lh_cuneus',           'lh_entorhinal',
    'lh_medialorbitofrontal', 'lh_pericalcarine'
]

RQ1_BETAS = {
    'lh_lateraloccipital':    -0.037,
    'rh_lateraloccipital':    -0.033,
    'lh_lingual':             -0.033,
    'rh_lingual':             -0.038,
    'lh_cuneus':              -0.041,
    'lh_entorhinal':          +0.089,
    'lh_medialorbitofrontal': -0.041,
    'lh_pericalcarine':       -0.038,
}

print("Setup completed.")
print(f"RQ1 confirmed regions: {len(RQ1_REGIONS)}")
"""

sources[3] = """## Block 2 — Data Preparation: 3 Pipelines

**Pipeline 1 — Raw:** ComBat harmonized data, no age correction  
**Pipeline 2 — Residualized:** Age effect removed using the regression model of the control group  
**Pipeline 3 — Standardized:** StandardScaler applied to both versions
"""

sources[4] = """df = pd.read_csv('abide_harmonised_combat.csv')
THICKNESS_COLS = [c for c in df.columns if c.startswith('lh_') or c.startswith('rh_')]

df_ctrl = df[df['DX_LABEL'] == 'Control'].copy().reset_index(drop=True)
df_asd  = df[df['DX_LABEL'] == 'ASD'].copy().reset_index(drop=True)

print(f"Data loaded:")
print(f"  ASD     : {len(df_asd)} subjects | age {df_asd['AGE_AT_SCAN'].min():.1f}–{df_asd['AGE_AT_SCAN'].max():.1f}")
print(f"  Control : {len(df_ctrl)} subjects | age {df_ctrl['AGE_AT_SCAN'].min():.1f}–{df_ctrl['AGE_AT_SCAN'].max():.1f}")
print(f"  Regions : {len(THICKNESS_COLS)}")
print(f"  ADOS (ASD): {df_asd['ADOS_TOTAL'].notna().sum()} / {len(df_asd)}")
"""

sources[5] = """# ── Pipeline 1: Raw (ComBat harmonized, no age correction) ──
age_ctrl = df_ctrl['AGE_AT_SCAN'].values.reshape(-1, 1)
age_asd  = df_asd['AGE_AT_SCAN'].values.reshape(-1, 1)

X_ham_68  = StandardScaler().fit_transform(df_asd[THICKNESS_COLS].values)
X_ham_8   = StandardScaler().fit_transform(df_asd[RQ1_REGIONS].values)

# ── Pipeline 2: Age Residualization ──
residuals_asd = np.zeros((len(df_asd), len(THICKNESS_COLS)))
r2_per_region = []

for i, col in enumerate(THICKNESS_COLS):
    model = LinearRegression().fit(age_ctrl, df_ctrl[col].values)
    residuals_asd[:, i] = df_asd[col].values - model.predict(age_asd)
    ss_res = np.sum((df_ctrl[col].values - model.predict(age_ctrl))**2)
    ss_tot = np.sum((df_ctrl[col].values - df_ctrl[col].mean())**2)
    r2_per_region.append(1 - ss_res / ss_tot)

r2_arr = np.array(r2_per_region)

rq1_idx = [THICKNESS_COLS.index(r) for r in RQ1_REGIONS]
X_res_68  = StandardScaler().fit_transform(residuals_asd)
X_res_8   = StandardScaler().fit_transform(residuals_asd[:, rq1_idx])

print("Pipelines ready:")
print(f"  Raw   68 regions: {X_ham_68.shape}")
print(f"  Raw    8 regions: {X_ham_8.shape}")
print(f"  Resid 68 regions: {X_res_68.shape}")
print(f"  Resid  8 regions: {X_res_8.shape}")
print(f"\nAge model (control group R²):")
print(f"  Mean R²     : {r2_arr.mean():.4f}  → age explains ~{r2_arr.mean()*100:.1f}% of thickness")
print(f"  Highest R²  : {r2_arr.max():.4f}  ({THICKNESS_COLS[r2_arr.argmax()]})")
print(f"  Lowest R²   : {r2_arr.min():.4f}  ({THICKNESS_COLS[r2_arr.argmin()]})")
print(f"\nAge-correlation change:")
r_before = np.mean([abs(stats.pearsonr(df_asd['AGE_AT_SCAN'], df_asd[c])[0])
                    for c in THICKNESS_COLS])
r_after  = np.mean([abs(stats.pearsonr(df_asd['AGE_AT_SCAN'], residuals_asd[:,i])[0])
                    for i in range(len(THICKNESS_COLS))])
print(f"  Before |r(thickness, age)| = {r_before:.4f}")
print(f"  After  |r(residual,  age)| = {r_after:.4f}  → {(1-r_after/r_before)*100:.1f}% reduction")
"""

sources[6] = """## Block 3 — 68 Regions vs 8 Regions: Clustering Comparison

K-Means (k=2) is run for both feature sets.  
Metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, Stability ARI, Age correlation
"""

sources[7] = """def run_clustering(X, label, k=2, n_boot=100):
    \"\"\"Applies K-Means, calculates metrics and stability.\"\"\"
    km = KMeans(n_clusters=k, n_init=30, random_state=SEED)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)
    db  = davies_bouldin_score(X, labels)

    # Bootstrap stability
    aris = []
    for b in range(n_boot):
        rng = np.random.RandomState(b)
        idx = rng.choice(len(X), size=int(0.8*len(X)), replace=False)
        l_b = KMeans(n_clusters=k, n_init=10, random_state=b).fit_predict(X[idx])
        aris.append(adjusted_rand_score(labels[idx], l_b))

    # Age correlation
    r_age, p_age = stats.pearsonr(labels, df_asd['AGE_AT_SCAN'].values)

    # Cluster sizes
    sizes = [(labels==c).sum() for c in range(k)]

    return {
        'label': label, 'labels': labels,
        'Silhouette': sil, 'CH': ch, 'DB': db,
        'ARI_mean': np.mean(aris), 'ARI_std': np.std(aris),
        'r_age': r_age, 'p_age': p_age,
        'sizes': sizes, 'X': X
    }

configs = {
    'Raw · 68 regions':    run_clustering(X_ham_68,  'Raw · 68 regions'),
    'Raw · 8 regions':     run_clustering(X_ham_8,   'Raw · 8 regions'),
    'Resid · 68 regions':  run_clustering(X_res_68,  'Resid · 68 regions'),
    'Resid · 8 regions':   run_clustering(X_res_8,   'Resid · 8 regions'),
}

print(f"{'Configuration':22s} {'Sil':>8} {'CH':>8} {'DB':>8} {'ARI':>8} {'r(age)':>9} {'p(age)':>10} {'Sizes'}")
print("─" * 90)
for name, res in configs.items():
    age_flag = ' ⚠️' if abs(res['r_age']) > 0.30 else ' ✅'
    print(f"{name:22s} {res['Silhouette']:>8.4f} {res['CH']:>8.1f} {res['DB']:>8.3f} "
          f"{res['ARI_mean']:>8.4f} {res['r_age']:>9.4f} {res['p_age']:>10.4f}"
          f"{age_flag}  {res['sizes']}")
"""

sources[8] = """# Visualization — 4 configurations comparison
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
cmap_age = plt.cm.plasma
c_colors = ['#2196F3', '#FF7043']

for col_i, (name, res) in enumerate(configs.items()):
    pca = PCA(n_components=2, random_state=SEED)
    Xp  = pca.fit_transform(res['X'])
    lbl = res['labels']
    age = df_asd['AGE_AT_SCAN'].values

    # Row 1: cluster color
    ax_top = axes[0, col_i]
    for c in [0, 1]:
        m = lbl == c
        ax_top.scatter(Xp[m,0], Xp[m,1], c=c_colors[c], alpha=0.45,
                       s=28, edgecolors='black', linewidth=0.2,
                       label=f'Cluster {c} (n={m.sum()})')
        # Mark the center
        ax_top.scatter(Xp[m,0].mean(), Xp[m,1].mean(), c=c_colors[c],
                       s=180, marker='*', edgecolors='black', linewidth=1, zorder=5)
        age_lbl = str(round(age[m].mean(), 0)) + 'y'
        ax_top.annotate(age_lbl,
                        (Xp[m,0].mean(), Xp[m,1].mean()),
                        textcoords='offset points', xytext=(6,6),
                        fontsize=9, fontweight='bold', color=c_colors[c])
    sil_v = res['Silhouette']; ari_v = res['ARI_mean']
    title_str = name + ' Sil=' + f'{sil_v:.3f}' + ' ARI=' + f'{ari_v:.3f}'
    ax_top.set_title(title_str,
                     fontweight='bold', fontsize=10)
    ax_top.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax_top.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax_top.legend(fontsize=8); ax_top.grid(True, alpha=0.2)

    # Row 2: age color
    ax_bot = axes[1, col_i]
    sc = ax_bot.scatter(Xp[:,0], Xp[:,1], c=age, cmap=cmap_age,
                        alpha=0.65, s=28, edgecolors='black', linewidth=0.2)
    plt.colorbar(sc, ax=ax_bot, label='Age', shrink=0.85)
    r_str = f'r={res["r_age"]:.3f}'
    color_r = 'darkred' if abs(res['r_age']) > 0.30 else 'darkgreen'
    ax_bot.set_title(f'Age Color | {r_str}', fontweight='bold',
                     fontsize=10, color=color_r)
    ax_bot.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax_bot.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax_bot.grid(True, alpha=0.2)

plt.suptitle('Block 3 — 68 Regions vs 8 Regions: Cluster Structure & Age Effect '
             '(Top row: cluster color · Bottom row: age color · Yellow=old, Purple=young)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('deep_blok3_clustering_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nInterpretation:")
print("  ⚠️ = r(cluster, age) > 0.30 → clustering captures the age effect")
print("  ✅ = r(cluster, age) < 0.30 → clustering is independent of age")
"""

sources[9] = """## Block 4 — PCA Loading Analysis

**PC1 loadings:** Which regions contribute the most to PC1 (separation axis)?

**Hypothesis 1 (Global Thickness):** Loadings values of all regions are in the same direction and close to each other  
→ PC1 only answers the question "is it overall thick or thin?"

**Hypothesis 2 (Regional Subtype):** Some regions are strongly positive, some are strongly negative  
→ PC1 represents a thickness *contrast* — specific regions are thick, others are thin

**Calculated separately for 4 configurations.**
"""

sources[10] = """def pca_loading_analysis(X, feature_names, config_name):
    \"\"\"PCA loading analysis — for H1 vs H2 testing.\"\"\"
    pca = PCA(n_components=min(10, len(feature_names)), random_state=SEED)
    pca.fit(X)

    # PC1 loadings
    pc1_load = pca.components_[0]
    pc2_load = pca.components_[1]

    # H1 test: how much in the same direction are the loadings?
    # If H1 is true, all should be positive or all negative
    same_sign_frac = max(
        (pc1_load > 0).sum() / len(pc1_load),
        (pc1_load < 0).sum() / len(pc1_load)
    )
    # Loadings std: low std = all similar magnitude = H1
    load_std = pc1_load.std()
    load_range = pc1_load.max() - pc1_load.min()

    # Strongest positive and negative contributors
    top_pos = np.argsort(pc1_load)[::-1][:5]
    top_neg = np.argsort(pc1_load)[:5]

    return {
        'pca': pca,
        'pc1_load': pc1_load,
        'pc2_load': pc2_load,
        'feature_names': feature_names,
        'same_sign_frac': same_sign_frac,
        'load_std': load_std,
        'load_range': load_range,
        'top_pos': top_pos,
        'top_neg': top_neg,
        'var_exp': pca.explained_variance_ratio_,
        'config_name': config_name
    }

# Feature names for 68 regions
feat_68 = THICKNESS_COLS
# Feature names for 8 regions
feat_8  = RQ1_REGIONS

loading_results = {
    'Raw · 68 regions':   pca_loading_analysis(X_ham_68,  feat_68, 'Raw · 68 regions'),
    'Raw · 8 regions':    pca_loading_analysis(X_ham_8,   feat_8,  'Raw · 8 regions'),
    'Resid · 68 regions': pca_loading_analysis(X_res_68,  feat_68, 'Resid · 68 regions'),
    'Resid · 8 regions':  pca_loading_analysis(X_res_8,   feat_8,  'Resid · 8 regions'),
}

print("PC1 Loading Analysis — H1 vs H2 Evaluation")
print("="*75)
print(f"{'Configuration':22s} {'Same Sign%':>10} {'Load Std':>10} {'Load Range':>12} {'Verdict'}")
print("─"*75)
for name, res in loading_results.items():
    h1 = res['same_sign_frac'] > 0.80 and res['load_std'] < 0.08
    verdict = "→ H1 (Global)" if h1 else "→ H2 (Regional)"
    print(f"{name:22s} {res['same_sign_frac']*100:>9.1f}% {res['load_std']:>10.4f} "
          f"{res['load_range']:>12.4f}  {verdict}")
"""

sources[11] = """# Loading visualization — all regions
fig, axes = plt.subplots(2, 2, figsize=(22, 16))
axes = axes.ravel()

for ax_i, (name, res) in enumerate(loading_results.items()):
    ax = axes[ax_i]
    loads = res['pc1_load']
    fnames = res['feature_names']
    n_feat = len(fnames)

    # Color: positive=red, negative=blue
    colors_bar = ['#c62828' if v > 0 else '#1565c0' for v in loads]
    sorted_idx = np.argsort(loads)

    bars = ax.barh(range(n_feat), loads[sorted_idx],
                   color=[colors_bar[i] for i in sorted_idx],
                   edgecolor='black', linewidth=0.3, height=0.7)
    ax.set_yticks(range(n_feat))
    labels_sorted = [fnames[i].replace('lh_','L·').replace('rh_','R·')
                     for i in sorted_idx]
    ax.set_yticklabels(labels_sorted, fontsize=7 if n_feat > 20 else 9)
    ax.axvline(0, color='black', linewidth=1.5)

    # H1/H2 verdict box
    same_pct = res['same_sign_frac'] * 100
    verdict = "H1: Global Thickness" if same_pct > 80 else "H2: Regional Contrast"
    verdict_color = '#e8f5e9' if same_pct > 80 else '#fff3e0'
    ax.text(0.98, 0.02, f"{verdict} ({same_pct:.0f}% same direction)",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=verdict_color,
                      edgecolor='grey', alpha=0.95))

    ax.set_xlabel('PC1 Loading Value', fontsize=10)
    var0 = res['var_exp'][0]; var1 = res['var_exp'][1]
    ax.set_title(f'{name} PC1 = {var0:.1%} variance | '
                 f'PC2 = {res["var_exp"][1]:.1%} variance',
                 fontweight='bold', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Block 4 — PC1 Loading Analysis: Which Regions Drive the Separation? '
             'Red = Positive contribution · Blue = Negative contribution '
             'All in the same direction → H1 (Global) | Mixed → H2 (Regional)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('deep_blok4_pca_loadings.png', dpi=150, bbox_inches='tight')
plt.show()
"""

sources[12] = """# Most critical configuration: Resid 8 regions (best clustering)
print("═"*65)
print("FOCUS: Resid · 8 Regions — PC1 Loading Detail")
print("═"*65)
res = loading_results['Resid · 8 regions']
loads = res['pc1_load']
fnames = res['feature_names']

print(f"\nPC1 explained variance: {res['var_exp'][0]:.1%}")
print(f"PC2 explained variance: {res['var_exp'][1]:.1%}")
print()
print(f"{'Region':35s} {'PC1 Loading':>12} {'PC2 Loading':>12} {'RQ1 β':>8}")
print("─"*70)
for i, name in enumerate(fnames):
    beta = RQ1_BETAS.get(name, 0)
    marker = '◀ STRONGEST' if abs(loads[i]) == max(abs(loads)) else ''
    print(f"{name:35s} {loads[i]:>12.4f} {res['pc2_load'][i]:>12.4f} "
          f"{beta:>8.3f}  {marker}")

print()
same_pct = res['same_sign_frac'] * 100
print(f"Same direction loading ratio: {same_pct:.1f}%")
if same_pct > 80:
    print("→ H1 is supported: PC1 represents global cortical thickness")
    print("  (One cluster is overall thicker, the other is overall thinner)")
else:
    print("→ H2 is supported: PC1 represents a regional contrast")
    print("  (Some regions are thick while others are thin — real subtype profile)")

# Entorhinal special case
entorhinal_load = loads[fnames.index('lh_entorhinal')]
print(f"\nSpecial case — lh_entorhinal:")
print(f"  PC1 loading = {entorhinal_load:.4f}")
print(f"  RQ1 beta    = +0.089 (thicker in ASD)")
print(f"  {'Expected direction ✅' if entorhinal_load > 0 else 'Opposite direction ⚠️'}")
"""

sources[13] = """## Block 5 — H1 vs H2 Hypothesis Testing

**H1 (Global Thickness Difference):** The difference between the two clusters is driven by global cortical thickness.  
Test: Is the *mean cortical thickness* of Cluster 0 and Cluster 1 significantly different?

**H2 (Regional Subtype):** The two clusters show differences in opposite directions in specific regions.  
Test: Are some regions thicker in Cluster 0 while thinner in Cluster 1?

**Dissociation index:** If high, H2; if low, H1.
"""

sources[14] = """# Best configuration: Resid 8 regions
best_labels = configs['Resid · 8 regions']['labels']
best_X      = configs['Resid · 8 regions']['X']

df_asd_clust = df_asd.copy()
df_asd_clust['Cluster'] = best_labels

print("H1 vs H2 HYPOTHESIS TESTING")
print("Configuration: Residualized · 8 RQ1 Regions · K-Means k=2")
print("="*65)

# H1 Test: Mean cortical thickness difference
mean_thick_c0 = df_asd_clust[df_asd_clust['Cluster']==0][THICKNESS_COLS].mean(axis=1)
mean_thick_c1 = df_asd_clust[df_asd_clust['Cluster']==1][THICKNESS_COLS].mean(axis=1)
t_h1, p_h1 = stats.ttest_ind(mean_thick_c0, mean_thick_c1)
d_h1 = (mean_thick_c0.mean() - mean_thick_c1.mean()) / np.sqrt(
    (mean_thick_c0.std()**2 + mean_thick_c1.std()**2) / 2)

print(f"\nH1 — Global Thickness Difference Test:")
print(f"  Cluster 0 mean thickness: {mean_thick_c0.mean():.4f} mm")
print(f"  Cluster 1 mean thickness: {mean_thick_c1.mean():.4f} mm")
print(f"  t={t_h1:.3f}, p={p_h1:.6f}, Cohen's d={d_h1:.4f}")
h1_sig = p_h1 < 0.05
print(f"  {'→ H1 SUPPORTED ✅' if h1_sig else '→ H1 NOT SUPPORTED ✗'}")

# H2 Test: Regional dissociation — are regions in different directions?
print(f"\nH2 — Regional Dissociation Test:")
region_diffs = []
dissociation_pairs = []
for col in RQ1_REGIONS:
    v0 = df_asd_clust[df_asd_clust['Cluster']==0][col]
    v1 = df_asd_clust[df_asd_clust['Cluster']==1][col]
    t, p = stats.ttest_ind(v0, v1)
    d = (v0.mean() - v1.mean()) / np.sqrt((v0.std()**2 + v1.std()**2) / 2)
    region_diffs.append({'Region': col, 't': t, 'p': p, 'd': d,
                         'mean_c0': v0.mean(), 'mean_c1': v1.mean()})

df_rdiff = pd.DataFrame(region_diffs).sort_values('d', ascending=False)
print(f"\n{'Region':35s} {'d(C0-C1)':>10} {'p':>10} {'C0 mean':>10} {'C1 mean':>10}")
print("─"*75)
for _, row in df_rdiff.iterrows():
    sig = '*' if row['p'] < 0.05 else ''
    direction = '↑C0' if row['d'] > 0 else '↓C0'
    print(f"{row['Region']:35s} {row['d']:>10.4f} {row['p']:>10.4f}  "
          f"{row['mean_c0']:>8.4f}  {row['mean_c1']:>8.4f}  {direction}{sig}")

# Dissociation index: high if some regions are + and others -
signs = np.sign(df_rdiff['d'].values)
dissociation_idx = 1 - abs(signs.mean())  # 0=global, 1=fully regional
print(f"\nDissociation index: {dissociation_idx:.4f}  (0=global, 1=fully regional)")
h2_verdict = "H2 SUPPORTED ✅" if dissociation_idx > 0.3 else "H2 NOT SUPPORTED ✗"
print(f"→ {h2_verdict}")
"""

sources[15] = """# H1 vs H2 visualization
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Top left: Mean thickness distribution (H1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(mean_thick_c0, bins=25, alpha=0.6, color='#2196F3',
         label=f'Cluster 0 (n={(best_labels==0).sum()})', edgecolor='black', linewidth=0.3)
ax1.hist(mean_thick_c1, bins=25, alpha=0.6, color='#FF7043',
         label=f'Cluster 1 (n={(best_labels==1).sum()})', edgecolor='black', linewidth=0.3)
ax1.axvline(mean_thick_c0.mean(), color='#2196F3', linewidth=2.5, linestyle='--')
ax1.axvline(mean_thick_c1.mean(), color='#FF7043', linewidth=2.5, linestyle='--')
ax1.set_xlabel('Mean Cortical Thickness (mm)')
ax1.set_ylabel('Number of Subjects')
ax1_title = 'H1 - Global: p=' + f'{p_h1:.4f}' + ' d=' + f'{d_h1:.4f}' + ('\n✅ Supported' if h1_sig else '\n✗ Not Supported')
ax1.set_title(ax1_title, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Top middle: Regional Cohen's d (H2)
ax2 = fig.add_subplot(gs[0, 1])
d_vals = df_rdiff['d'].values
region_labels = [r.replace('lh_','L-').replace('rh_','R-')
                 for r in df_rdiff['Region'].values]
bar_colors = ['#c62828' if d > 0 else '#1565c0' for d in d_vals]
bars = ax2.barh(range(len(d_vals)), d_vals, color=bar_colors,
                edgecolor='black', linewidth=0.5, height=0.6)
ax2.set_yticks(range(len(d_vals)))
ax2.set_yticklabels(region_labels, fontsize=9)
ax2.axvline(0, color='black', linewidth=1.5)
ax2.set_xlabel("Cohen's d (Cluster 0 − Cluster 1)")
ax2.set_title(f'H2 — Regional Dissociation idx={dissociation_idx:.3f} '
              f'{"✅ Supported" if dissociation_idx > 0.3 else "✗ Not Supported"}',
              fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Top right: Cluster profile heatmap
ax3 = fig.add_subplot(gs[0, 2])
cent_data = []
for c in [0, 1]:
    m = best_labels == c
    cent_data.append(best_X[m].mean(axis=0))
cent_df = pd.DataFrame(cent_data,
                       columns=[r.replace('lh_','L-').replace('rh_','R-')
                                for r in RQ1_REGIONS],
                       index=['Cluster 0', 'Cluster 1'])
sns.heatmap(cent_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            linewidths=1.0, ax=ax3,
            cbar_kws={'label': 'Z-score'})
ax3.set_title('Cluster Centroid Profiles (Standardized Residual)',
              fontweight='bold')

# Bottom left: Lollipop plot — value of each region in the two clusters
ax4 = fig.add_subplot(gs[1, :2])
x  = np.arange(len(RQ1_REGIONS))
y0 = [df_asd_clust[df_asd_clust['Cluster']==0][r].mean() for r in RQ1_REGIONS]
y1 = [df_asd_clust[df_asd_clust['Cluster']==1][r].mean() for r in RQ1_REGIONS]

ax4.scatter(x, y0, c='#2196F3', s=100, zorder=5, label='Cluster 0', edgecolors='black', linewidth=0.5)
ax4.scatter(x, y1, c='#FF7043', s=100, zorder=5, label='Cluster 1', edgecolors='black', linewidth=0.5)
for xi, v0, v1 in zip(x, y0, y1):
    ax4.plot([xi, xi], [v0, v1], color='grey', linewidth=1.5, alpha=0.6, linestyle='--')
ax4.set_xticks(x)
ax4.set_xticklabels([r.replace('lh_','L-').replace('rh_','R-') for r in RQ1_REGIONS],
                    rotation=30, ha='right', fontsize=9)
ax4.set_ylabel('Mean Thickness (mm)')
ax4.set_title('Cluster 0 vs Cluster 1 — Region-based Thickness Profiles\nLines show difference magnitude', fontweight='bold')
ax4.legend(); ax4.grid(axis='y', alpha=0.3)

# Bottom right: Scatter — PC1 vs global thickness
ax5 = fig.add_subplot(gs[1, 2])
pca_best = PCA(n_components=2, random_state=SEED)
Xp_best  = pca_best.fit_transform(best_X)
for c, col in [(0, '#2196F3'), (1, '#FF7043')]:
    m = best_labels == c
    ax5.scatter(Xp_best[m,0],
                df_asd_clust[m][THICKNESS_COLS].mean(axis=1),
                c=col, alpha=0.4, s=25, edgecolors='black', linewidth=0.2,
                label=f'Cluster {c}')
r_pc1_global, p_pc1 = stats.pearsonr(
    Xp_best[:, 0],
    df_asd_clust[THICKNESS_COLS].mean(axis=1)
)
ax5.set_xlabel('PC1 Score')
ax5.set_ylabel('Mean Global Thickness (mm)')
ax5.set_title(f'PC1 vs Global Thickness r={r_pc1_global:.4f}, p={p_pc1:.4f}\n'
              f'{"High r → PC1 = global scale (H1)" if abs(r_pc1_global)>0.5 else "Low r → PC1 ≠ global scale"}',
              fontweight='bold')
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.2)

plt.suptitle('Block 5 — H1 (Global Thickness) vs H2 (Regional Subtype) Hypothesis Testing',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig('deep_blok5_h1_h2_test.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nResult summary:")
print(f"  H1 (Global): p={p_h1:.4f}  {'✅' if h1_sig else '✗'}")
print(f"  H2 (Regional dissociation index): {dissociation_idx:.4f}  "
      f"{'✅' if dissociation_idx>0.3 else '✗'}")
print(f"  PC1 ↔ Global thickness correlation: r={r_pc1_global:.4f}")
"""

sources[16] = """## Block 6 — LDA: Maximum Discriminative Axis

PCA maximizes variance — not the separation between clusters.  
**LDA (Linear Discriminant Analysis)** finds the linear combination that precisely separates the clusters.

LDA loading values = "how much does this region contribute to separating the two clusters?"
"""

sources[17] = """# LDA — for 68 regions and 8 regions
results_lda = {}

for feat_name, X, fnames in [
    ('Raw · 68 regions',   X_ham_68,  THICKNESS_COLS),
    ('Raw · 8 regions',    X_ham_8,   RQ1_REGIONS),
    ('Resid · 68 regions', X_res_68,  THICKNESS_COLS),
    ('Resid · 8 regions',  X_res_8,   RQ1_REGIONS),
]:
    lbl = configs[feat_name]['labels']
    lda = LDA(n_components=1)
    lda.fit(X, lbl)
    lda_scores = lda.transform(X).ravel()
    coefs = lda.coef_[0]

    # LDA separation quality
    t, p = stats.ttest_ind(lda_scores[lbl==0], lda_scores[lbl==1])
    d = (lda_scores[lbl==0].mean() - lda_scores[lbl==1].mean()) / np.sqrt(
        (lda_scores[lbl==0].std()**2 + lda_scores[lbl==1].std()**2) / 2)

    results_lda[feat_name] = {
        'coefs': coefs, 'fnames': fnames,
        'lda_scores': lda_scores, 'labels': lbl,
        't': t, 'p': p, 'd': d
    }

print("LDA Separation Quality:")
print(f"{'Configuration':22s} {'t-stat':>10} {'p-value':>12} {'Cohen d':>10}")
print("─"*58)
for name, res in results_lda.items():
    print(f"{name:22s} {res['t']:>10.3f} {res['p']:>12.6f} {res['d']:>10.4f}")
"""

sources[18] = """# LDA loading visualization for the two most critical configurations
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

plot_configs = ['Raw · 8 regions', 'Resid · 8 regions',
                'Raw · 68 regions', 'Resid · 68 regions']

for ax_i, cname in enumerate(plot_configs):
    res = results_lda[cname]
    coefs = res['coefs']
    fnames = res['fnames']
    n = len(fnames)
    ax = axes[ax_i // 2][ax_i % 2]

    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    top_n = min(20, n)
    top_idx = sorted_idx[:top_n]

    bar_vals = coefs[top_idx]
    bar_cols = ['#c62828' if v > 0 else '#1565c0' for v in bar_vals]
    bar_labels = [fnames[i].replace('lh_','L-').replace('rh_','R-') for i in top_idx]

    ax.barh(range(top_n), bar_vals[::-1],
            color=bar_cols[::-1], edgecolor='black', linewidth=0.4, height=0.65)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(bar_labels[::-1], fontsize=8 if n > 20 else 10)
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_xlabel('LDA Coefficient (larger = more discriminative)')
    ax.set_title(f'LDA — {cname} '
                 f't={res["t"]:.2f}, p={res["p"]:.4f}, d={res["d"]:.3f}',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Highlight the strongest region
    strongest = fnames[sorted_idx[0]].replace('lh_','L-').replace('rh_','R-')
    ax.text(0.98, 0.02, f'Strongest: {strongest}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow',
                                  edgecolor='grey', alpha=0.9))

plt.suptitle('Block 6 — LDA: Brain Regions Maximally Separating the Clusters\n'
             'Red = Points to Cluster 0 - Blue = Points to Cluster 1\n'
             'Magnitude = Discriminative power',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('deep_blok6_lda_loadings.png', dpi=150, bbox_inches='tight')
plt.show()
"""

sources[19] = """# LDA score distribution for the best configuration
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
c_colors = ['#2196F3', '#FF7043']

for ax_i, cname in enumerate(['Raw · 8 regions', 'Resid · 8 regions']):
    res = results_lda[cname]
    ax = axes[ax_i]
    for c in [0, 1]:
        m = res['labels'] == c
        ax.hist(res['lda_scores'][m], bins=25, alpha=0.6,
                color=c_colors[c], label=f'Cluster {c} (n={m.sum()})',
                edgecolor='black', linewidth=0.3)
        ax.axvline(res['lda_scores'][m].mean(), color=c_colors[c],
                   linewidth=2.5, linestyle='--')
    ax.set_xlabel('LDA Score (Discriminative Axis)')
    ax.set_ylabel('Number of Subjects')
    age_r, _ = stats.pearsonr(res['lda_scores'],
                               df_asd['AGE_AT_SCAN'].values)
    ax.set_title(f'{cname} t={res["t"]:.2f}, p={res["p"]:.6f}, d={res["d"]:.3f}\n'
                 f'r(LDA, age) = {age_r:.4f}  '
                 f'{"⚠️ age effect" if abs(age_r)>0.30 else "✅ independent of age"}',
                 fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Block 6 — LDA Score Distribution: Raw vs Residualized',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('deep_blok6_lda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# Relation between LDA and age
print("LDA Score and Age Correlation:")
for cname, res in results_lda.items():
    r, p = stats.pearsonr(res['lda_scores'], df_asd['AGE_AT_SCAN'].values)
    flag = '⚠️ Age effect present' if abs(r) > 0.30 else '✅ Independent of age'
    print(f"  {cname:22s}: r={r:.4f}, p={p:.6f}  {flag}")
"""

sources[20] = """## Block 7 — Cluster Profile Analysis: What do the Two Groups Represent Anatomically?

For the best configuration (Residualized · 8 regions):
- Anatomical profile of Cluster 0 and Cluster 1
- Statistical difference in each region
- Grouping by lobe
- Association with clinical variables (ADOS, IQ)
"""

sources[21] = """best_labels = configs['Resid · 8 regions']['labels']
df_asd_clust = df_asd.copy()
df_asd_clust['Cluster'] = best_labels

# ── Anatomical region based details ──
print("CLUSTER ANATOMICAL PROFILE — Residualized · 8 Regions · K-Means k=2")
print("="*75)
print()

for c in [0, 1]:
    sub = df_asd_clust[df_asd_clust['Cluster'] == c]
    print(f"CLUSTER {c} (n={len(sub)}):")
    print(f"  Age: {sub['AGE_AT_SCAN'].mean():.1f} ± {sub['AGE_AT_SCAN'].std():.1f} years")
    print(f"  Male: {100*(sub['SEX_LABEL']=='Male').mean():.1f}%")
    print(f"  FIQ: {sub['FIQ'].mean():.1f} ± {sub['FIQ'].std():.1f}")
    print(f"  VIQ: {sub['VIQ'].mean():.1f}  |  PIQ: {sub['PIQ'].mean():.1f}")
    ados_sub = sub['ADOS_TOTAL'].dropna()
    print(f"  ADOS: {ados_sub.mean():.2f} ± {ados_sub.std():.2f} (n={len(ados_sub)})")
    print()
    print(f"  Cortical thickness (raw value, mm):")
    for reg in RQ1_REGIONS:
        vals = sub[reg]
        print(f"    {reg:35s}: {vals.mean():.4f} ± {vals.std():.4f}")
    print()

# Statistical tests
print("STATISTICAL TESTS:")
print("-"*50)
for var, label in [('AGE_AT_SCAN','Age'),('FIQ','FIQ'),('VIQ','VIQ'),
                   ('PIQ','PIQ'),('ADOS_TOTAL','ADOS')]:
    g0 = df_asd_clust[df_asd_clust['Cluster']==0][var].dropna()
    g1 = df_asd_clust[df_asd_clust['Cluster']==1][var].dropna()
    if len(g0)>=2 and len(g1)>=2:
        t, p = stats.ttest_ind(g0, g1)
        d = (g0.mean()-g1.mean())/np.sqrt((g0.std()**2+g1.std()**2)/2)
        sig = '*** p<0.001' if p<0.001 else '** p<0.01' if p<0.01 else '* p<0.05' if p<0.05 else 'ns'
        print(f"  {label:8s}: t={t:.3f}, p={p:.4f}, d={d:.3f}  {sig}")
ct = pd.crosstab(df_asd_clust['Cluster'], df_asd_clust['SEX_LABEL'])
chi2, p_sex, _, _ = stats.chi2_contingency(ct)
print(f"  Gender: χ²={chi2:.3f}, p={p_sex:.4f}  {'* p<0.05' if p_sex<0.05 else 'ns'}")
"""

sources[22] = """# Anatomik görselleştirme -> # Anatomical visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
c_colors = ['#2196F3', '#FF7043']

# Region-based profile
ax = axes[0, 0]
x  = np.arange(len(RQ1_REGIONS))
y0 = [df_asd_clust[df_asd_clust['Cluster']==0][r].mean() for r in RQ1_REGIONS]
y1 = [df_asd_clust[df_asd_clust['Cluster']==1][r].mean() for r in RQ1_REGIONS]
w  = 0.35
ax.bar(x - w/2, y0, w, color='#2196F3', alpha=0.8, label='Cluster 0', edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, y1, w, color='#FF7043', alpha=0.8, label='Cluster 1', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([r.replace('lh_','L').replace('rh_','R') for r in RQ1_REGIONS],
                   rotation=35, ha='right', fontsize=8)
ax.set_ylabel('Mean Thickness (mm)')
ax.set_title('Region-based Thickness Profile', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# ADOS boxplot
ax = axes[0, 1]
ados_data = [df_asd_clust[df_asd_clust['Cluster']==c]['ADOS_TOTAL'].dropna().values
             for c in [0, 1]]
bp = ax.boxplot(ados_data, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2))
for patch, col in zip(bp['boxes'], c_colors):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax.set_xticklabels(['Cluster 0 n=' + str(len(ados_data[0])),
                    'Cluster 1 n=' + str(len(ados_data[1]))])
g0a = df_asd_clust[df_asd_clust['Cluster']==0]['ADOS_TOTAL'].dropna()
g1a = df_asd_clust[df_asd_clust['Cluster']==1]['ADOS_TOTAL'].dropna()
t_a, p_a = stats.ttest_ind(g0a, g1a)
ados_title = 'ADOS Distribution  t=' + f'{t_a:.2f}' + '  p=' + f'{p_a:.4f}'
ax.set_title(ados_title, fontweight='bold')
ax.set_ylabel('ADOS Total Score')
ax.grid(axis='y', alpha=0.3)

# IQ profile
ax = axes[0, 2]
iq_vars = ['FIQ', 'VIQ', 'PIQ']
x_iq = np.arange(len(iq_vars))
for c_i, c in enumerate([0, 1]):
    sub = df_asd_clust[df_asd_clust['Cluster']==c]
    means = [sub[v].mean() for v in iq_vars]
    stds  = [sub[v].std()  for v in iq_vars]
    ax.errorbar(x_iq + c_i*0.2 - 0.1, means, yerr=stds, fmt='o-',
                color=c_colors[c_i], linewidth=2, markersize=8, capsize=4,
                label=f'Cluster {c}')
ax.set_xticks(x_iq)
ax.set_xticklabels(iq_vars)
ax.set_ylabel('IQ Score')
ax.set_title('IQ Profile (Mean +/- SD)', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# Age violin
ax = axes[1, 0]
parts = ax.violinplot([df_asd_clust[df_asd_clust['Cluster']==c]['AGE_AT_SCAN'].values
                       for c in [0, 1]],
                      positions=[0, 1], showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(c_colors[i]); pc.set_alpha(0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Cluster 0 n=' + str((best_labels==0).sum()),
                    'Cluster 1 n=' + str((best_labels==1).sum())])
g0y = df_asd_clust[df_asd_clust['Cluster']==0]['AGE_AT_SCAN']
g1y = df_asd_clust[df_asd_clust['Cluster']==1]['AGE_AT_SCAN']
t_y, p_y = stats.ttest_ind(g0y, g1y)
yas_title = 'Age Distribution  t=' + f'{t_y:.2f}' + '  p=' + f'{p_y:.4f}'
yas_title += ('  (Significant AGE DIFFERENCE)' if p_y < 0.05 else '  (Independent of Age)')
ax.set_title(yas_title, fontweight='bold')
ax.set_ylabel('Age (years)')
ax.grid(axis='y', alpha=0.3)

# PC1 vs ADOS scatter
ax = axes[1, 1]
pca_plot = PCA(n_components=2, random_state=SEED)
Xp_plot  = pca_plot.fit_transform(configs['Resid · 8 regions']['X'])
for c in [0, 1]:
    m = best_labels == c
    has_ados = m & df_asd_clust['ADOS_TOTAL'].notna().values
    ax.scatter(Xp_plot[has_ados, 0], df_asd_clust[has_ados]['ADOS_TOTAL'],
               c=c_colors[c], alpha=0.4, s=25, label=f'Cluster {c}',
               edgecolors='black', linewidth=0.2)
has_ados_all = df_asd_clust['ADOS_TOTAL'].notna().values
r_pc1_ados, p_pc1_ados = stats.pearsonr(
    Xp_plot[has_ados_all, 0], df_asd_clust[has_ados_all]['ADOS_TOTAL'])
ax.set_xlabel('PC1 Score'); ax.set_ylabel('ADOS Total')
pc1ados_title = 'PC1 vs ADOS  r=' + f'{r_pc1_ados:.4f}' + '  p=' + f'{p_pc1_ados:.4f}'
ax.set_title(pc1ados_title, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Site distribution
ax = axes[1, 2]
ct_site = pd.crosstab(df_asd_clust['Cluster'], df_asd_clust['SITE_ID'],
                      normalize='index') * 100
top_sites = df_asd_clust['SITE_ID'].value_counts().head(8).index
ct_site[top_sites].T.plot(kind='bar', ax=ax,
                           color=c_colors, edgecolor='black', width=0.7)
ax.set_xlabel('Site'); ax.set_ylabel('%')
ax.set_title('Site Distribution (Top 8 sites)', fontweight='bold')
ax.legend(fontsize=9); ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Block 7 - Cluster Profile Analysis: Who are the Two Groups?',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('deep_blok7_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.show()
"""

sources[23] = """## Block 8 — Pipeline Effect Measurement

We measure the contribution of harmonization and age correction to clustering quality.

**Measurement criteria:**
- Silhouette (cluster quality ↑)
- Bootstrap ARI (stability ↑)
- r(cluster, age) (age dependency ↓ → closer to 0 is better)
- r(cluster, site) (site dependency ↓ → closer to 0 is better)
"""

sources[24] = """# Calculate site correlation for each configuration as well
from sklearn.preprocessing import LabelEncoder

le_site = LabelEncoder()
site_encoded = le_site.fit_transform(df_asd['SITE_ID'].values)

pipeline_summary = {}
for name, res in configs.items():
    lbl = res['labels']
    r_age,  p_age  = stats.pearsonr(lbl, df_asd['AGE_AT_SCAN'].values)
    r_site, p_site = stats.pointbiserialr(lbl, site_encoded)
    pipeline_summary[name] = {
        'Silhouette':  res['Silhouette'],
        'ARI':         res['ARI_mean'],
        '|r_age|':     abs(r_age),
        '|r_site|':    abs(r_site),
        'p_age':       p_age,
        'p_site':      p_site,
    }

print("PIPELINE EFFECT SUMMARY TABLE")
print("="*80)
print(f"{'Configuration':22s} {'Sil↑':>8} {'ARI↑':>8} {'|r_age|↓':>10} {'|r_site|↓':>11}")
print("─"*65)
for name, res in pipeline_summary.items():
    age_ok  = '✅' if res['|r_age|']  < 0.15 else '⚠️'
    site_ok = '✅' if res['|r_site|'] < 0.10 else '⚠️'
    print(f"{name:22s} {res['Silhouette']:>8.4f} {res['ARI']:>8.4f} "
          f"{res['|r_age|']:>8.4f} {age_ok}  {res['|r_site|']:>9.4f} {site_ok}")

print()
print("Interpretation:")
print("  |r_age|  < 0.15 → sufficiently independent of age ✅")
print("  |r_site| < 0.10 → sufficiently independent of site effect ✅")
"""

sources[25] = """# Visualization — pipeline comparison bar/radar
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

names_list = list(pipeline_summary.keys())
sil_vals   = [pipeline_summary[n]['Silhouette'] for n in names_list]
ari_vals   = [pipeline_summary[n]['ARI']        for n in names_list]
r_age_vals = [pipeline_summary[n]['|r_age|']    for n in names_list]
r_site_vals= [pipeline_summary[n]['|r_site|']   for n in names_list]

x = np.arange(len(names_list))
w = 0.2

bars1 = axes[0].bar(x - 1.5*w, sil_vals,   w, label='Silhouette ↑', color='#2196F3',
                     edgecolor='black', linewidth=0.5)
bars2 = axes[0].bar(x - 0.5*w, ari_vals,   w, label='Stability ARI ↑', color='#4CAF50',
                     edgecolor='black', linewidth=0.5)
bars3 = axes[0].bar(x + 0.5*w, r_age_vals, w, label='|r_age| ↓', color='#FF9800',
                     edgecolor='black', linewidth=0.5)
bars4 = axes[0].bar(x + 1.5*w, r_site_vals,w, label='|r_site| ↓', color='#E91E63',
                     edgecolor='black', linewidth=0.5)

axes[0].set_xticks(x)
axes[0].set_xticklabels(names_list, rotation=20, ha='right', fontsize=9)
axes[0].set_ylabel('Value')
axes[0].set_title('Pipeline Comparison — 4 Metrics\n'
                  'Silhouette & ARI should be high | |r_age| & |r_site| should be low',
                  fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].axhline(0.15, color='#FF9800', linestyle=':', alpha=0.5, linewidth=1.5)
axes[0].axhline(0.10, color='#E91E63', linestyle=':', alpha=0.5, linewidth=1.5)
axes[0].grid(axis='y', alpha=0.3)

# "Best configuration" score — composite
# composite = Sil*0.35 + ARI*0.35 - |r_age|*0.15 - |r_site|*0.15
composite = [
    pipeline_summary[n]['Silhouette'] * 0.35 + pipeline_summary[n]['ARI']        * 0.35 -
    pipeline_summary[n]['|r_age|']    * 0.15 - pipeline_summary[n]['|r_site|']   * 0.15
    for n in names_list ]
bar_colors = ['#4CAF50' if v == max(composite) else '#BDBDBD' for v in composite]
axes[1].bar(names_list, composite, color=bar_colors, edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Composite Score')
axes[1].set_xticklabels(names_list, rotation=20, ha='right', fontsize=9)
axes[1].set_title('Composite Pipeline Score\n'
                  '0.35×Sil + 0.35×ARI − 0.15×|r_age| − 0.15×|r_site|',
                  fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

best_name = names_list[np.argmax(composite)]
axes[1].text(np.argmax(composite), max(composite) + 0.005,
             '← Best', ha='center', fontsize=10, fontweight='bold', color='darkgreen')

plt.suptitle('Block 8 — Pipeline Effect: Contribution of Harmonization & Age Correction',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('deep_blok8_pipeline_effect.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nBest configuration (composite): {best_name}")
print(f"Composite score: {max(composite):.4f}")
"""

sources[26] = """## Block 9 — General Summary & Presentation-Ready Findings"""

sources[27] = """print("╔" + "═"*70 + "╗")
print("║  GENERAL SUMMARY — RQ2 DEEP ANALYSIS                                ║")
print("╠" + "═"*70 + "╣")

print("║                                                                      ║")
print("║  BEST CONFIGURATION:                                                 ║")
print("║  Residualized (age removed) · 8 RQ1 Regions · K-Means k=2            ║")
print("║                                                                      ║")

best_res = configs['Resid · 8 regions']
best_pipe = pipeline_summary['Resid · 8 regions']

print(f"║  Silhouette  : {best_res['Silhouette']:.4f}  (0=poor, 1=perfect)               ║")
print(f"║  Stability   : {best_res['ARI_mean']:.4f}  (ARI, 100× bootstrap)               ║")
print(f"║  |r(age)|    : {best_pipe['|r_age|']:.4f}  (0.07 → independent of age ✅)      ║")
print(f"║  |r(site)|   : {best_pipe['|r_site|']:.4f}  (ComBat effective ✅)              ║")
print("║                                                                      ║")
print("╠" + "═"*70 + "╣")
print("║  H1 vs H2 RESULT:                                                    ║")

if h1_sig:
    print(f"║  H1 (Global thickness difference): SUPPORTED  p={p_h1:.4f} d={d_h1:.3f}     ║")
else:
    print(f"║  H1 (Global thickness difference): ✗ NOT SUPPORTED  p={p_h1:.4f}           ║")

if dissociation_idx > 0.3:
    print(f"║  H2 (Regional subtype)     : SUPPORTED  idx={dissociation_idx:.3f}             ║")
else:
    print(f"║  H2 (Regional subtype)     : ✗ NOT SUPPORTED  idx={dissociation_idx:.3f}          ║")

print(f"║  PC1 ↔ Global thickness      : r={r_pc1_global:.4f}                           ║")
print("║                                                                      ║")
print("╠" + "═"*70 + "╣")
print("║  CLINICAL VALIDATION:                                                ║")

g0a = df_asd_clust[df_asd_clust['Cluster']==0]['ADOS_TOTAL'].dropna()
g1a = df_asd_clust[df_asd_clust['Cluster']==1]['ADOS_TOTAL'].dropna()
t_ados, p_ados = stats.ttest_ind(g0a, g1a)

print(f"║  ADOS Cluster 0: {g0a.mean():.2f} ± {g0a.std():.2f}  (n={len(g0a)})                       ║")
print(f"║  ADOS Cluster 1: {g1a.mean():.2f} ± {g1a.std():.2f}  (n={len(g1a)})                       ║")
print(f"║  ADOS diff  : p={p_ados:.4f}  → {'Significant ✅' if p_ados<0.05 else 'NOT significant ✗'}                       ║")
print("║                                                                      ║")
print("╠" + "═"*70 + "╣")
print("║  ANATOMICAL INTERPRETATION:                                          ║")
print("║                                                                      ║")

# Which hypothesis is supported?
if abs(r_pc1_global) > 0.5:
    print("║  PC1 shows a strong correlation with global cortical thickness.      ║")
    print("║  → Cluster separation is largely driven by GLOBAL THICKNESS diff.    ║")
    print("║  → H1 dominant                                                       ║")
else:
    print("║  PC1 shows a weak correlation with global cortical thickness.        ║")
    print("║  → Cluster separation is driven by REGIONAL contrast profile.        ║")
    print("║  → H2 dominant                                                       ║")

print("║                                                                      ║")
print("║  Strongest LDA regions (Resid · 8 regions):                          ║")
lda_res = results_lda['Resid · 8 regions']
top3_idx = np.argsort(np.abs(lda_res['coefs']))[::-1][:3]
for rank, idx in enumerate(top3_idx):
    reg = lda_res['fnames'][idx]
    coef = lda_res['coefs'][idx]
    print(f"║    {rank+1}. {reg:35s} LDA={coef:+.4f}          ║")

print("║                                                                      ║")
print("╠" + "═"*70 + "╣")
print("║  PIPELINE CONTRIBUTION:                                              ║")
print("║  Raw 68 regions  → r_age=0.48 ⚠️  Silhouette=0.233                   ║")
print("║  + Age correct   → r_age=0.07 ✅  Silhouette=0.171 (noise increased) ║")
print("║  + 8 RQ1 regions → r_age=0.07 ✅  Silhouette=0.269 ← BEST            ║")
print("║  → Harmonization cleared the site effect                             ║")
print("║  → Age correction removed the age confound                           ║")
print("║  → RQ1 region selection reduced noise and increased SNR              ║")
print("║    ratio                                                             ║")
print("╚" + "═"*70 + "╝")
"""

# Reconstruct cell sources with line-breaks
for idx, source_text in sources.items():
    # Convert string to list of lines with newline character at the end
    lines = [line + '\n' for line in source_text.split('\n')]
    if lines and lines[-1] == '\n':
        lines = lines[:-1]  # Remove trailing empty newline if it is extra
    elif lines:
        # Strip the last newline if the original text did not end with it,
        # but Jupyter cells typically have newlines except the last character.
        # It's cleaner to keep them as is.
        pass
    
    # Clean up empty strings or fix last element newlines
    lines_cleaned = []
    for i, line in enumerate(lines):
        if i == len(lines) - 1:
            # Last line shouldn't have trailing \n if we want to be clean, but either works.
            lines_cleaned.append(line.rstrip('\r\n'))
        else:
            lines_cleaned.append(line)
            
    nb['cells'][idx]['source'] = lines_cleaned
    # Clear outputs and execution counts to make it clean
    if nb['cells'][idx]['cell_type'] == 'code':
        nb['cells'][idx]['outputs'] = []
        nb['cells'][idx]['execution_count'] = None

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook successfully translated and saved to:", output_path)
