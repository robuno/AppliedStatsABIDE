
# --- Cell 2 --- 

# Standard scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling tools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# Statistics
from scipy import stats
from scipy.stats import f as f_dist

import warnings
warnings.filterwarnings('ignore')

# Visualization defaults
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.titleweight'] = 'bold'

# Reproducibility
RNG = 42
np.random.seed(RNG)

print('Imports OK')

# --- Cell 4 --- 

# Load the three datasets
df_clean = pd.read_csv('../dataset/abide_solution_A.csv')
df_harm  = pd.read_csv('abide_harmonised_combat.csv')
df_pheno = pd.read_csv('../Phenotypic_V1_0b_preprocessed1.csv')

print(f'df_clean shape : {df_clean.shape}')
print(f'df_harm shape  : {df_harm.shape}')
print(f'df_pheno shape : {df_pheno.shape}')

print('\nFirst few columns of df_harm:')
print(list(df_harm.columns[:15]))

# --- Cell 5 --- 

# Auto-detect the 68 cortical thickness columns by prefix
THICK_COLS = [c for c in df_harm.columns if c.startswith('lh_') or c.startswith('rh_')]
print(f'Number of thickness columns detected: {len(THICK_COLS)}')
print(f'  Left hemisphere : {sum(c.startswith("lh_") for c in THICK_COLS)} regions')
print(f'  Right hemisphere: {sum(c.startswith("rh_") for c in THICK_COLS)} regions')

# Sanity check: df_clean must have the same thickness columns
clean_thick = [c for c in df_clean.columns if c.startswith('lh_') or c.startswith('rh_')]
assert set(THICK_COLS) == set(clean_thick), 'Mismatch between df_clean and df_harm thickness columns'

# Safety: drop any rows with missing thickness values (df_clean / df_harm should already be clean)
n_before_clean = len(df_clean)
n_before_harm  = len(df_harm)
df_clean = df_clean.dropna(subset=THICK_COLS).reset_index(drop=True)
df_harm  = df_harm.dropna(subset=THICK_COLS).reset_index(drop=True)
print(f'\nDropped {n_before_clean - len(df_clean)} rows from df_clean and '
      f'{n_before_harm - len(df_harm)} rows from df_harm (missing thickness)')

# Ensure the diagnosis label column exists
for d in (df_clean, df_harm):
    if 'DX_LABEL' not in d.columns:
        d['DX_LABEL'] = d['DX_GROUP'].map({1: 'ASD', 2: 'Control'})

print(f'\nFinal sizes — clean: {df_clean.shape}, harmonized: {df_harm.shape}')
print('\nDiagnosis breakdown (harmonized):')
print(df_harm['DX_LABEL'].value_counts())
print(f'\nNumber of sites: {df_harm["SITE_ID"].nunique()}')

# --- Cell 7 --- 

# Fit a separate scaler for each dataset (they have different means/SDs)
scaler_raw  = StandardScaler()
scaler_harm = StandardScaler()

X_raw  = scaler_raw.fit_transform(df_clean[THICK_COLS])
X_harm = scaler_harm.fit_transform(df_harm[THICK_COLS])

print(f'X_raw  shape: {X_raw.shape},  mean ≈ {X_raw.mean():.2e}, std ≈ {X_raw.std():.3f}')
print(f'X_harm shape: {X_harm.shape}, mean ≈ {X_harm.mean():.2e}, std ≈ {X_harm.std():.3f}')
print('\nStandardization OK: both centered at 0 with unit variance')

# --- Cell 9 --- 

# PCA on the harmonized data — main analysis target
pca_full = PCA(n_components=len(THICK_COLS), random_state=RNG)
scores_full = pca_full.fit_transform(X_harm)

# PCA on the raw (unharmonized) data — for the before-after comparison only
pca_raw = PCA(n_components=len(THICK_COLS), random_state=RNG)
scores_raw = pca_raw.fit_transform(X_raw)

print('PCA fitted on harmonized data')
print(f'  Total variance captured by all 68 components: '
      f'{pca_full.explained_variance_ratio_.sum()*100:.2f}%')
print('\nFirst 10 components — individual and cumulative variance:')
print(f'  {"PC":>4} {"Indiv (%)":>10} {"Cumul (%)":>10}')
cum = 0
for i in range(10):
    v = pca_full.explained_variance_ratio_[i] * 100
    cum += v
    print(f'  {i+1:>4} {v:>10.2f} {cum:>10.2f}')

# --- Cell 11 --- 

exp_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

# Find k for each variance threshold
k_80 = int(np.argmax(cum_var >= 0.80) + 1)
k_90 = int(np.argmax(cum_var >= 0.90) + 1)
k_95 = int(np.argmax(cum_var >= 0.95) + 1)

print('Components needed to reach each variance threshold:')
print(f'   80% variance → k = {k_80}')
print(f'   90% variance → k = {k_90}')
print(f'   95% variance → k = {k_95}')
print(f'\nReduction ratio (80%): {k_80}/68 = {k_80/68*100:.1f}% of the original dimensions')

# --- Cell 12 --- 

# Two-panel scree plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left panel: individual variance for the first 20 components
n_show = 20
axes[0].bar(range(1, n_show + 1), exp_var[:n_show] * 100,
            color='steelblue', alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].plot(range(1, n_show + 1), exp_var[:n_show] * 100, 'o-',
             color='darkred', linewidth=2, markersize=6)
axes[0].set_xlabel('Principal Component', fontsize=11)
axes[0].set_ylabel('Variance Explained (%)', fontsize=11)
axes[0].set_title(f'Scree Plot — first {n_show} components')
axes[0].set_xticks(range(1, n_show + 1))
axes[0].grid(alpha=0.3)

# Right panel: cumulative variance for all 68 components
axes[1].plot(range(1, len(cum_var) + 1), cum_var * 100, 'o-',
             linewidth=2, markersize=4, color='darkgreen')
axes[1].fill_between(range(1, len(cum_var) + 1), 0, cum_var * 100,
                      color='darkgreen', alpha=0.15)
axes[1].axhline(80, color='orange', linestyle='--', linewidth=1.5, label='80%')
axes[1].axhline(90, color='red',    linestyle='--', linewidth=1.5, label='90%')
axes[1].axhline(95, color='purple', linestyle='--', linewidth=1.5, label='95%')
axes[1].axvline(k_80, color='orange', linestyle=':', alpha=0.7)
axes[1].axvline(k_90, color='red',    linestyle=':', alpha=0.7)
axes[1].axvline(k_95, color='purple', linestyle=':', alpha=0.7)
axes[1].annotate(f'k={k_80}', xy=(k_80, 80), xytext=(k_80 + 2, 65),
                 color='orange', fontsize=11, fontweight='bold')
axes[1].annotate(f'k={k_90}', xy=(k_90, 90), xytext=(k_90 + 2, 82),
                 color='red', fontsize=11, fontweight='bold')
axes[1].annotate(f'k={k_95}', xy=(k_95, 95), xytext=(k_95 + 2, 96.5),
                 color='purple', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 102)

plt.tight_layout()
plt.show()

# --- Cell 14 --- 

# Loadings matrix — columns = first 5 PCs, rows = brain regions
loadings = pd.DataFrame(
    pca_full.components_[:5].T,
    index=THICK_COLS,
    columns=[f'PC{i+1}' for i in range(5)]
)

fig, axes = plt.subplots(1, 2, figsize=(16, 9))

# PC1 loadings — should be predominantly positive (global thickness factor)
pc1_sorted = loadings['PC1'].sort_values()
colors_pc1 = ['steelblue' if x > 0 else 'darkred' for x in pc1_sorted]
axes[0].barh(range(len(pc1_sorted)), pc1_sorted.values,
             color=colors_pc1, alpha=0.85, edgecolor='black', linewidth=0.2)
axes[0].set_yticks(range(len(pc1_sorted)))
axes[0].set_yticklabels(pc1_sorted.index, fontsize=6)
axes[0].set_xlabel('PC1 Loading')
axes[0].set_title(f'PC1 Loadings — {pca_full.explained_variance_ratio_[0]*100:.1f}% variance\n'
                  f'(Global thickness factor)')
axes[0].axvline(0, color='black', linewidth=0.5)
axes[0].grid(alpha=0.3, axis='x')

# PC2 loadings — bipolar (occipital + vs temporal -)
pc2_sorted = loadings['PC2'].sort_values()
colors_pc2 = ['steelblue' if x > 0 else 'darkred' for x in pc2_sorted]
axes[1].barh(range(len(pc2_sorted)), pc2_sorted.values,
             color=colors_pc2, alpha=0.85, edgecolor='black', linewidth=0.2)
axes[1].set_yticks(range(len(pc2_sorted)))
axes[1].set_yticklabels(pc2_sorted.index, fontsize=6)
axes[1].set_xlabel('PC2 Loading')
axes[1].set_title(f'PC2 Loadings — {pca_full.explained_variance_ratio_[1]*100:.1f}% variance\n'
                  f'(Occipital + / Temporal -)')
axes[1].axvline(0, color='black', linewidth=0.5)
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Summary tables of the polarity at each end of PC2
print('\nTop 5 regions with POSITIVE PC2 loading (occipital / visual cortex expected):')
print(loadings['PC2'].sort_values(ascending=False).head(5).to_frame().round(3))
print('\nTop 5 regions with NEGATIVE PC2 loading (temporal / memory cortex expected):')
print(loadings['PC2'].sort_values().head(5).to_frame().round(3))

# Quantify how positive PC1 is on average — should be near-uniformly positive
pc1_share_positive = (loadings['PC1'] > 0).mean() * 100
print(f'\nPC1 — share of regions with positive loading: {pc1_share_positive:.1f}% '
      f'(higher = stronger global thickness interpretation)')

# --- Cell 16 --- 

# To keep MDS tractable, use a random subsample for visualization
N_MDS = min(600, X_harm.shape[0])
rng = np.random.default_rng(RNG)
mds_idx = rng.choice(X_harm.shape[0], size=N_MDS, replace=False)

# Align both raw and harmonized data on the SAME subjects for a fair comparison
# (df_clean and df_harm may have slightly different row orderings — match by SUB_ID)
common_ids = set(df_clean['SUB_ID']) & set(df_harm['SUB_ID'])
df_clean_aligned = df_clean[df_clean['SUB_ID'].isin(common_ids)].sort_values('SUB_ID').reset_index(drop=True)
df_harm_aligned  = df_harm [df_harm ['SUB_ID'].isin(common_ids)].sort_values('SUB_ID').reset_index(drop=True)

X_raw_aligned  = scaler_raw.transform (df_clean_aligned[THICK_COLS])
X_harm_aligned = scaler_harm.transform(df_harm_aligned [THICK_COLS])
# Recompute PCA scores on the aligned harmonized data (will be near-identical to scores_full)
scores_full_aligned = pca_full.transform(X_harm_aligned)
scores_raw_aligned  = pca_raw .transform(X_raw_aligned)

# Resample the index against the aligned arrays
N_MDS = min(600, X_harm_aligned.shape[0])
mds_idx = rng.choice(X_harm_aligned.shape[0], size=N_MDS, replace=False)

X_harm_sub  = X_harm_aligned[mds_idx]
X_raw_sub   = X_raw_aligned [mds_idx]
labels_dx   = df_harm_aligned['DX_LABEL'].iloc[mds_idx].values
labels_site = df_harm_aligned['SITE_ID'].iloc[mds_idx].values

# PCA-reduced (top k_80 components)
X_harm_pca = scores_full_aligned[mds_idx, :k_80]
X_raw_pca  = scores_raw_aligned [mds_idx, :k_80]

print(f'MDS sample size: {N_MDS} subjects')
print(f'Raw 68D shape  : {X_harm_sub.shape}')
print(f'PCA-reduced    : {X_harm_pca.shape}  (k={k_80})')

def run_mds(X, n_components=2, random_state=RNG):
    """Lightweight MDS fit — fewer inits + capped iterations for speed."""
    mds = MDS(
        n_components=n_components,
        random_state=random_state,
        n_init=1,
        max_iter=200,
        normalized_stress='auto',
        n_jobs=-1,
    )
    return mds.fit_transform(X)

# --- Cell 17 --- 

# Run all four MDS embeddings — this takes a few minutes
print('Running MDS — raw 68D, UN-harmonized...')
mds_raw_full  = run_mds(X_raw_sub)
print('Running MDS — raw 68D, harmonized...')
mds_harm_full = run_mds(X_harm_sub)
print(f'Running MDS — PCA k={k_80}, UN-harmonized...')
mds_raw_pca   = run_mds(X_raw_pca)
print(f'Running MDS — PCA k={k_80}, harmonized...')
mds_harm_pca  = run_mds(X_harm_pca)
print('All four MDS fits done.')

# --- Cell 18 --- 

# Plot — 2 rows (color by site / by diagnosis) × 4 columns (configurations)
fig, axes = plt.subplots(2, 4, figsize=(22, 11))

sites = sorted(np.unique(labels_site))
site_palette = sns.color_palette('tab20', n_colors=len(sites))
dx_palette = {'ASD': '#d62728', 'Control': '#1f77b4'}

configs = [
    (mds_raw_full,  'Raw 68D — UN-harmonized'),
    (mds_harm_full, 'Raw 68D — Harmonized'),
    (mds_raw_pca,   f'PCA k={k_80} — UN-harmonized'),
    (mds_harm_pca,  f'PCA k={k_80} — Harmonized'),
]

# Top row: colour by SITE_ID
for col, (emb, title) in enumerate(configs):
    ax = axes[0, col]
    for i, site in enumerate(sites):
        m = labels_site == site
        ax.scatter(emb[m, 0], emb[m, 1], c=[site_palette[i]],
                   label=site, s=18, alpha=0.7, edgecolors='none')
    ax.set_title(f'{title}\nColored by SITE', fontsize=10)
    ax.set_xlabel('MDS-1'); ax.set_ylabel('MDS-2')
    if col == 3:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=7, ncol=1, frameon=False)

# Bottom row: colour by diagnosis
for col, (emb, title) in enumerate(configs):
    ax = axes[1, col]
    for dx in ['Control', 'ASD']:
        m = labels_dx == dx
        ax.scatter(emb[m, 0], emb[m, 1], c=dx_palette[dx],
                   label=dx, s=18, alpha=0.55, edgecolors='none')
    ax.set_title(f'{title}\nColored by DIAGNOSIS', fontsize=10)
    ax.set_xlabel('MDS-1'); ax.set_ylabel('MDS-2')
    ax.legend(loc='best', fontsize=9)

plt.suptitle('MDS — Does site or diagnosis structure the data?', fontsize=13, y=1.00)
plt.tight_layout()
plt.show()

# --- Cell 21 --- 

def hotelling_t2_two_sample(X1, X2):
    """Two-sample Hotelling's T² test.

    Parameters
    ----------
    X1, X2 : 2D ndarrays of shape (n1, p) and (n2, p)

    Returns
    -------
    dict with keys: T2, F, p_value, df1, df2, n1, n2, p
    Uses the Moore-Penrose pseudo-inverse for numerical stability in high
    dimensions or near-singular covariance.
    """
    n1, p = X1.shape
    n2, p2 = X2.shape
    assert p == p2, 'Dimension mismatch between groups'

    mean_diff = X1.mean(axis=0) - X2.mean(axis=0)
    S1 = np.cov(X1, rowvar=False, ddof=1)
    S2 = np.cov(X2, rowvar=False, ddof=1)
    S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    S_inv = np.linalg.pinv(S_pooled)   # pseudo-inverse: safe for any condition number

    T2 = (n1 * n2 / (n1 + n2)) * mean_diff @ S_inv @ mean_diff

    df1 = p
    df2 = n1 + n2 - p - 1
    if df2 <= 0:
        return {'T2': float(T2), 'F': np.nan, 'p_value': np.nan,
                'df1': df1, 'df2': df2, 'n1': n1, 'n2': n2, 'p': p}

    F = ((n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)) * T2
    p_value = float(1 - f_dist.cdf(F, df1, df2))

    return {'T2': float(T2), 'F': float(F), 'p_value': p_value,
            'df1': int(df1), 'df2': int(df2),
            'n1': int(n1), 'n2': int(n2), 'p': int(p)}


# Sanity check on synthetic data — two groups shifted in mean should give large F
np.random.seed(0)
A = np.random.randn(80, 5)
B = np.random.randn(80, 5) + 0.5
sanity = hotelling_t2_two_sample(A, B)
print(f'Sanity check (synthetic, mean shift = 0.5):  F = {sanity["F"]:.2f},  p = {sanity["p_value"]:.4e}')
print('-> Both values should reflect a strong, highly significant difference.')

# --- Cell 22 --- 

# Boolean masks for ASD and Control
mask_asd = (df_harm['DX_LABEL'] == 'ASD').values
mask_con = (df_harm['DX_LABEL'] == 'Control').values

print(f'n_ASD     = {mask_asd.sum()}')
print(f'n_Control = {mask_con.sum()}')

# Hotelling's T² on the FULL 68-dimensional harmonized space
result_full = hotelling_t2_two_sample(X_harm[mask_asd], X_harm[mask_con])

print('\n' + '=' * 60)
print("Hotelling's T² — FULL 68D harmonized space")
print('=' * 60)
print(f'  T²      = {result_full["T2"]:>10.3f}')
print(f'  F-stat  = {result_full["F"]:>10.3f}    (df₁ = {result_full["df1"]}, df₂ = {result_full["df2"]})')
print(f'  p-value = {result_full["p_value"]:>10.3e}')
sig = 'SIGNIFICANT' if result_full['p_value'] < 0.05 else 'not significant'
print(f'  ASD vs Control multivariate difference: {sig} at α = 0.05')

# --- Cell 23 --- 

# Sweep — run Hotelling's T² across many values of k, plus at the reduced 80% point
ks = list(range(2, 69))
sweep = []
for k in ks:
    Z = scores_full[:, :k]
    res = hotelling_t2_two_sample(Z[mask_asd], Z[mask_con])
    res['k'] = k
    sweep.append(res)

df_sweep = pd.DataFrame(sweep)

# Identify the maximum-F point
idx_max = df_sweep['F'].idxmax()
best_k     = int(df_sweep.loc[idx_max, 'k'])
best_F     = float(df_sweep.loc[idx_max, 'F'])
best_T2    = float(df_sweep.loc[idx_max, 'T2'])
best_p     = float(df_sweep.loc[idx_max, 'p_value'])

# Get T² and F at k_80 specifically
row_k80 = df_sweep[df_sweep['k'] == k_80].iloc[0]
F_at_k80 = float(row_k80['F'])
T2_at_k80 = float(row_k80['T2'])
p_at_k80 = float(row_k80['p_value'])

print('Hotelling sweep — selected k values:')
print(df_sweep[df_sweep['k'].isin([2, 5, 10, k_80, k_90, 40, 60, 68])]
      [['k', 'T2', 'F', 'p_value']].round(4).to_string(index=False))

print(f'\n>>> Best F-statistic across sweep: k = {best_k},  F = {best_F:.3f},  p = {best_p:.3e}')
print(f'>>> F at k₈₀ ({k_80} components): F = {F_at_k80:.3f},  p = {p_at_k80:.3e}')
print(f'>>> F at full 68D:                F = {result_full["F"]:.3f},  p = {result_full["p_value"]:.3e}')

if best_k < 68 and best_F > result_full['F']:
    print('\n[RESULT] PCA reduction CONCENTRATES the signal: peak F is at reduced dimension.')
elif abs(best_F - result_full['F']) < 0.2:
    print('\n[RESULT] PCA reduction is NEUTRAL: F-statistic is essentially flat across k.')
else:
    print('\n[RESULT] Full 68D retains the strongest signal.')

# --- Cell 24 --- 

# Plot the Hotelling sweep — T², F, and -log10(p) vs k
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1 — T² vs k (grows monotonically with k as more dimensions feed the test)
axes[0].plot(df_sweep['k'], df_sweep['T2'], 'o-', color='steelblue', linewidth=2, markersize=4)
axes[0].axvline(k_80, color='orange', linestyle='--', linewidth=1.5, label=f'k₈₀ = {k_80}')
axes[0].axvline(best_k, color='green', linestyle=':', linewidth=1.5, label=f'argmax F at k = {best_k}')
axes[0].set_xlabel('Number of PCA components (k)')
axes[0].set_ylabel("Hotelling's T²")
axes[0].set_title("T² vs k")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Panel 2 — F-statistic vs k (penalizes added dimensions; this is the test that matters)
axes[1].plot(df_sweep['k'], df_sweep['F'], 'o-', color='darkgreen', linewidth=2, markersize=4)
axes[1].axvline(k_80, color='orange', linestyle='--', linewidth=1.5, label=f'k₈₀ = {k_80}')
axes[1].axvline(best_k, color='green', linestyle=':', linewidth=1.5, label=f'argmax at k = {best_k}')
axes[1].axhline(result_full['F'], color='gray', linestyle='-.', linewidth=1.5,
                label=f'Full 68D F = {result_full["F"]:.2f}')
axes[1].set_xlabel('Number of PCA components (k)')
axes[1].set_ylabel('F-statistic')
axes[1].set_title('F-statistic vs k')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Panel 3 — significance vs k
log_p = -np.log10(np.clip(df_sweep['p_value'].values, 1e-300, None))
axes[2].plot(df_sweep['k'], log_p, 'o-', color='darkred', linewidth=2, markersize=4)
axes[2].axhline(-np.log10(0.05), color='red', linestyle=':', linewidth=1.5, label='α = 0.05')
axes[2].axvline(k_80, color='orange', linestyle='--', linewidth=1.5, label=f'k₈₀ = {k_80}')
axes[2].axvline(best_k, color='green', linestyle=':', linewidth=1.5, label=f'argmax at k = {best_k}')
axes[2].axhline(-np.log10(result_full['p_value']), color='gray', linestyle='-.', linewidth=1.5,
                label='Full 68D')
axes[2].set_xlabel('Number of PCA components (k)')
axes[2].set_ylabel('−log₁₀(p-value)')
axes[2].set_title('Significance vs k')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle("Hotelling's T² across PCA dimensions — does compression help or hurt?",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# --- Cell 27 --- 

# Per-PC two-sample t-test (Welch) + Cohen's d
pc_rows = []
for i in range(scores_full.shape[1]):
    asd_scores = scores_full[mask_asd, i]
    con_scores = scores_full[mask_con, i]
    t_stat, p_val = stats.ttest_ind(asd_scores, con_scores, equal_var=False)
    # Pooled-SD Cohen's d
    pooled_sd = np.sqrt((asd_scores.var(ddof=1) + con_scores.var(ddof=1)) / 2)
    d = (asd_scores.mean() - con_scores.mean()) / pooled_sd if pooled_sd > 0 else 0
    pc_rows.append({
        'PC': i + 1,
        'var_explained_%': pca_full.explained_variance_ratio_[i] * 100,
        'mean_ASD': asd_scores.mean(),
        'mean_Control': con_scores.mean(),
        'cohens_d': d,
        't_stat': t_stat,
        'p_value': p_val,
    })

df_pc = pd.DataFrame(pc_rows)

n_sig    = int((df_pc['p_value'] < 0.05).sum())
n_strong = int((df_pc['p_value'] < 0.01).sum())
print(f'PCs significant at p < 0.05 : {n_sig} / {len(df_pc)}')
print(f'PCs significant at p < 0.01 : {n_strong} / {len(df_pc)}')

# Show the 15 most discriminative PCs
print('\nTop 15 most discriminative PCs (sorted by p-value):')
print(df_pc.sort_values('p_value').head(15).round(4).to_string(index=False))

# --- Cell 28 --- 

# Visualize the per-PC results
fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))

# Panel 1 — −log10(p) for every PC, coloured by significance
log_p_pc = -np.log10(np.clip(df_pc['p_value'].values, 1e-300, None))
bar_colors = ['darkred' if p < 0.01 else ('orange' if p < 0.05 else 'lightgray')
              for p in df_pc['p_value']]
axes[0].bar(df_pc['PC'], log_p_pc, color=bar_colors, edgecolor='black', linewidth=0.3, alpha=0.85)
axes[0].axhline(-np.log10(0.05), color='orange', linestyle='--', linewidth=1.5, label='p = 0.05')
axes[0].axhline(-np.log10(0.01), color='darkred', linestyle='--', linewidth=1.5, label='p = 0.01')
axes[0].set_xlabel('PC number')
axes[0].set_ylabel('−log₁₀(p-value)')
axes[0].set_title('Per-PC ASD vs Control significance (all 68 PCs)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Panel 2 — top 20 PCs by |Cohen's d|
df_top_d = (df_pc.assign(abs_d=df_pc['cohens_d'].abs())
                  .sort_values('abs_d', ascending=False)
                  .head(20))
colors_d = ['darkred' if d < 0 else 'steelblue' for d in df_top_d['cohens_d']]
axes[1].barh(range(len(df_top_d)), df_top_d['cohens_d'], color=colors_d, alpha=0.85,
             edgecolor='black', linewidth=0.3)
axes[1].set_yticks(range(len(df_top_d)))
axes[1].set_yticklabels([f'PC{int(p)}' for p in df_top_d['PC']])
axes[1].invert_yaxis()
axes[1].axvline(0, color='black', linewidth=0.5)
axes[1].set_xlabel("Cohen's d (ASD − Control)")
axes[1].set_title("Top 20 PCs by |Cohen's d|")
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# --- Cell 29 --- 

# Score-distribution plots for the 4 most discriminative PCs
top4_pcs = df_pc.sort_values('p_value').head(4)['PC'].values

fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
for i, pc in enumerate(top4_pcs):
    ax = axes[i]
    asd_scores = scores_full[mask_asd, pc - 1]
    con_scores = scores_full[mask_con, pc - 1]
    sns.kdeplot(con_scores, ax=ax, color='#1f77b4', label='Control', fill=True, alpha=0.4)
    sns.kdeplot(asd_scores, ax=ax, color='#d62728', label='ASD',     fill=True, alpha=0.4)
    p_val = float(df_pc[df_pc['PC'] == pc]['p_value'].iloc[0])
    d_val = float(df_pc[df_pc['PC'] == pc]['cohens_d'].iloc[0])
    var_pct = float(df_pc[df_pc['PC'] == pc]['var_explained_%'].iloc[0])
    ax.set_title(f'PC{pc}  ({var_pct:.1f}% var)\np = {p_val:.2e},  d = {d_val:+.3f}')
    ax.set_xlabel(f'PC{pc} score')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Score distributions of the 4 most discriminative PCs', fontsize=12, y=1.04)
plt.tight_layout()
plt.show()

# --- Cell 30 --- 

# Connect significant PCs back to anatomy — show the top 5 absolute-loading regions
# for each of the top-5 most discriminative PCs
top5_pcs = df_pc.sort_values('p_value').head(5)['PC'].astype(int).values

print('Top 5 most discriminative PCs — anatomical interpretation')
print('=' * 78)
for pc in top5_pcs:
    pc_idx = pc - 1
    loads = pd.Series(pca_full.components_[pc_idx], index=THICK_COLS)
    p_val = float(df_pc[df_pc['PC'] == pc]['p_value'].iloc[0])
    d_val = float(df_pc[df_pc['PC'] == pc]['cohens_d'].iloc[0])
    var_pct = float(df_pc[df_pc['PC'] == pc]['var_explained_%'].iloc[0])

    print(f'\nPC{pc}  |  variance = {var_pct:5.2f}%  |  p = {p_val:.2e}  |  d = {d_val:+.3f}')
    print('  Top 5 |loading| regions:')
    top_load = loads.reindex(loads.abs().sort_values(ascending=False).index).head(5)
    for region, val in top_load.items():
        print(f'    {region:<35} {val:+.3f}')

# --- Cell 32 --- 

# Final summary table — everything in one place
print('=' * 78)
print('RQ3 SUMMARY — DIMENSIONALITY REDUCTION & SEPARABILITY')
print('=' * 78)

print('\n[1] PCA — variance compression')
print(f'    68 features    →  {k_80} components keep 80% of variance ({k_80/68*100:.1f}% of dims)')
print(f'                       {k_90} components keep 90% of variance')
print(f'                       {k_95} components keep 95% of variance')
print(f'    Top 3 PCs:        {exp_var[0]*100:.1f}% + {exp_var[1]*100:.1f}% + {exp_var[2]*100:.1f}% = '
      f'{(exp_var[0]+exp_var[1]+exp_var[2])*100:.1f}% variance')

print('\n[2] MDS visualization')
print('    - Site clusters dissolve after ComBat → harmonization confirmed visually')
print('    - ASD / Control clouds overlap in every 2D projection — global cortical')
print('      thickness is not the discriminative axis')

print("\n[3] Hotelling's T²")
print(f'    Full 68D    :  F = {result_full["F"]:.3f},  p = {result_full["p_value"]:.3e}')
print(f'    PCA k={k_80:<3}   :  F = {F_at_k80:.3f},  p = {p_at_k80:.3e}')
print(f'    argmax F    :  k = {best_k},  F = {best_F:.3f},  p = {best_p:.3e}')
delta_F = best_F - result_full['F']
print(f'    ΔF (argmax − full): {delta_F:+.3f}  '
      f'({"+++ reduction helps" if delta_F > 0.5 else "~~ neutral" if abs(delta_F) <= 0.5 else "--- reduction hurts"})')

print('\n[4] Per-PC univariate tests')
print(f'    PCs significant at p<0.05 : {n_sig}/{len(df_pc)}  '
      f'({n_sig/len(df_pc)*100:.1f}%)')
print(f'    PCs significant at p<0.01 : {n_strong}/{len(df_pc)}  '
      f'({n_strong/len(df_pc)*100:.1f}%)')
print(f'    Most discriminative PC    : PC{int(df_pc.sort_values("p_value").iloc[0]["PC"])}  '
      f'(p = {df_pc["p_value"].min():.3e})')

# Recommendation logic
print('\n' + '=' * 78)
print('RECOMMENDATION FOR RQ4')
print('=' * 78)

if best_k <= k_80 and best_F > result_full['F']:
    print(f'  → Use PCA scores with k ∈ [{best_k}, {k_80}] as input to the RQ4 classifier.')
    print(f'    Reduction concentrates the ASD signal AND reduces parameter count by ~{(1-k_80/68)*100:.0f}%.')
elif abs(best_F - result_full['F']) <= 0.5:
    print(f'  → PCA reduction is essentially neutral on the multivariate test.')
    print(f'    Either representation is valid for RQ4. Prefer k={k_80} for parsimony.')
else:
    print('  → Keep the full 68 features for RQ4 — reduction loses signal.')
