
# --- Cell 2 --- 

# ============================================================
# ImportsÜ
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# sklearn
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score,
    classification_report
)

# neuroCombat — required for proper in-fold harmonization
# pip install neuroCombat
from neuroCombat import neuroCombat, neuroCombatFromTraining

# Styling
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 110
RNG = 42
np.random.seed(RNG)

print('Setup complete.')


# --- Cell 3 --- 

# ============================================================
# Load datasets
# ============================================================
df_clean = pd.read_csv('../dataset/abide_solution_A.csv')
df_harm  = pd.read_csv('abide_harmonised_combat.csv')
df_pheno = pd.read_csv('../Phenotypic_V1_0b_preprocessed1.csv')

# Cortical thickness columns (68 regions: 34 lh + 34 rh)
THICKNESS_COLS = sorted([c for c in df_clean.columns if c.startswith(('lh_', 'rh_'))])
assert len(THICKNESS_COLS) == 68, f'Expected 68 thickness columns, got {len(THICKNESS_COLS)}'

# Ensure metadata columns are present in df_clean (merge from phenotypic if missing)
for col in ['SITE_ID', 'AGE_AT_SCAN', 'SEX', 'DX_GROUP']:
    if col not in df_clean.columns:
        df_clean = df_clean.merge(df_pheno[['SUB_ID', col]], on='SUB_ID', how='left')

# Drop subjects with any missing thickness / key metadata values (safety net)
df_clean = df_clean.dropna(subset=THICKNESS_COLS + ['SITE_ID', 'AGE_AT_SCAN', 'SEX', 'DX_GROUP']).reset_index(drop=True)

print(f'df_clean: {df_clean.shape}')
print(f'df_harm : {df_harm.shape}')
print(f'df_pheno: {df_pheno.shape}')
print(f'Thickness columns: {len(THICKNESS_COLS)}')
print(f'Sites in df_clean: {df_clean["SITE_ID"].nunique()}')


# --- Cell 5 --- 

# Class balance
print('Class balance (DX_GROUP: 1=ASD, 2=Control)')
print(df_clean['DX_GROUP'].value_counts().rename(index={1:'ASD',2:'Control'}))
print()

# Site × diagnosis
site_dx = pd.crosstab(df_clean['SITE_ID'], df_clean['DX_GROUP'].map({1:'ASD',2:'Control'}))
site_dx['Total'] = site_dx.sum(axis=1)
print('Subjects per site × diagnosis (top 10)')
print(site_dx.sort_values('Total', ascending=False).head(10))
print()

# Group-level demographics
demo = df_clean.groupby('DX_GROUP').agg(
    N=('SUB_ID', 'size'),
    Age_mean=('AGE_AT_SCAN', 'mean'),
    Age_std=('AGE_AT_SCAN', 'std'),
    Pct_Male=('SEX', lambda s: 100 * (s == 1).mean())
).round(2)
demo.index = demo.index.map({1: 'ASD', 2: 'Control'})
print('Demographics by diagnosis')
print(demo)


# --- Cell 6 --- 

# Three-panel EDA figure
fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))

# (1) Class balance
ax = axes[0]
counts = df_clean['DX_GROUP'].map({1:'ASD', 2:'Control'}).value_counts()
bars = ax.bar(counts.index, counts.values, color=['#d62728', '#1f77b4'])
ax.set_title('Class balance')
ax.set_ylabel('N subjects')
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 5, str(v), ha='center', fontweight='bold')

# (2) Age distribution
ax = axes[1]
for g, lbl, c in [(1, 'ASD', '#d62728'), (2, 'Control', '#1f77b4')]:
    sns.kdeplot(df_clean.loc[df_clean['DX_GROUP'] == g, 'AGE_AT_SCAN'],
                ax=ax, label=lbl, color=c, fill=True, alpha=0.3)
ax.set_title('Age distribution by diagnosis')
ax.set_xlabel('Age (years)')
ax.legend()

# (3) Site × diagnosis stacked horizontal bars
ax = axes[2]
sdx = pd.crosstab(df_clean['SITE_ID'], df_clean['DX_GROUP'].map({1:'ASD', 2:'Control'}))
sdx = sdx.loc[sdx.sum(axis=1).sort_values(ascending=True).index]
sdx.plot(kind='barh', stacked=True, ax=ax, color=['#d62728', '#1f77b4'])
ax.set_title('Subjects per site × diagnosis')
ax.set_xlabel('N subjects')
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()


# --- Cell 8 --- 

def fit_combat_train(X_train_df, sites_train, age_train, sex_train):
    '''Fit ComBat on training data.

    Parameters
    ----------
    X_train_df : DataFrame (n_samples, n_features)
    sites_train, age_train, sex_train : Series of length n_samples

    Returns
    -------
    X_harm : DataFrame, same shape as X_train_df, harmonized
    estimates : dict, ComBat parameter estimates (used to transform new data)
    '''
    # neuroCombat expects shape (n_features, n_samples)
    data_T = X_train_df.T.values
    covars = pd.DataFrame({
        'SITE_ID': sites_train.reset_index(drop=True).astype(str).values,
        'AGE':     age_train.reset_index(drop=True).values,
        'SEX':     sex_train.reset_index(drop=True).astype(str).values,
    })
    result = neuroCombat(
        dat=data_T,
        covars=covars,
        batch_col='SITE_ID',
        categorical_cols=['SEX'],
        continuous_cols=['AGE'],
    )
    X_harm = pd.DataFrame(
        result['data'].T,
        columns=X_train_df.columns,
        index=X_train_df.index,
    )
    return X_harm, result['estimates']


def apply_combat_test(X_test_df, sites_test, estimates):
    '''Apply pre-fit ComBat parameters to held-out data.'''
    data_T = X_test_df.T.values
    result = neuroCombatFromTraining(
        dat=data_T,
        batch=sites_test.reset_index(drop=True).astype(str).values,
        estimates=estimates,
    )
    X_harm = pd.DataFrame(
        result['data'].T,
        columns=X_test_df.columns,
        index=X_test_df.index,
    )
    return X_harm


def compute_metrics(y_true, y_pred, y_proba=None):
    '''Binary classification metrics. y_true: 1 = ASD (positive), 0 = Control.'''
    m = {
        'accuracy':    accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred, pos_label=1),   # ASD recall (clinical screening)
        'specificity': recall_score(y_true, y_pred, pos_label=0),   # Control recall
        'precision':   precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1':          f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if y_proba is not None:
        m['auc_roc'] = roc_auc_score(y_true, y_proba)
    return m


def cross_validate_pipeline(X, y, sites, age, sex, classifier,
                            harmonize=True, n_splits=5, random_state=RNG, verbose=False):
    '''Stratified k-fold CV with optional in-fold ComBat harmonization.

    Note: when `harmonize=True`, ComBat is fitted INSIDE every train fold and
    its parameters are applied to the validation fold (no leakage).
    '''
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics, fold_preds = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        s_tr, s_va = sites.iloc[tr_idx], sites.iloc[va_idx]
        a_tr, a_va = age.iloc[tr_idx], age.iloc[va_idx]
        x_tr, x_va = sex.iloc[tr_idx], sex.iloc[va_idx]

        # Sites in val but not in train → ComBat would fail; skip those subjects from val
        common_sites = set(s_tr.unique())
        keep_mask = s_va.isin(common_sites).values
        if not keep_mask.all():
            X_va = X_va.loc[keep_mask].reset_index(drop=True)
            y_va = y_va.loc[keep_mask].reset_index(drop=True)
            s_va = s_va.loc[keep_mask].reset_index(drop=True)
            a_va = a_va.loc[keep_mask].reset_index(drop=True)
            x_va = x_va.loc[keep_mask].reset_index(drop=True)

        if harmonize:
            X_tr_h, est = fit_combat_train(X_tr, s_tr, a_tr, x_tr)
            X_va_h = apply_combat_test(X_va, s_va, est)
        else:
            X_tr_h, X_va_h = X_tr, X_va

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_h)
        X_va_s = scaler.transform(X_va_h)

        clf = clone(classifier)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_va_s)
        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(X_va_s)[:, 1]
        elif hasattr(clf, 'decision_function'):
            y_proba = clf.decision_function(X_va_s)
        else:
            y_proba = None

        m = compute_metrics(y_va, y_pred, y_proba)
        m['fold'] = fold + 1
        fold_metrics.append(m)
        fold_preds.append({'fold': fold + 1, 'y_true': y_va.values,
                           'y_pred': y_pred, 'y_proba': y_proba})
        if verbose:
            print(f'  Fold {fold+1}: acc={m["accuracy"]:.3f}  sens={m["sensitivity"]:.3f}  '
                  f'spec={m["specificity"]:.3f}  auc={m.get("auc_roc", float("nan")):.3f}')

    return pd.DataFrame(fold_metrics), fold_preds


# --- Cell 10 --- 

# Re-encode diagnosis: 1 = ASD (positive), 0 = Control
y_all     = (df_clean['DX_GROUP'] == 1).astype(int)
X_all     = df_clean[THICKNESS_COLS].copy()
sites_all = df_clean['SITE_ID'].astype(str).copy()
age_all   = df_clean['AGE_AT_SCAN'].copy()
sex_all   = df_clean['SEX'].astype(str).copy()

print(f'Total subjects        : {len(df_clean)}')
print(f'ASD (label 1)         : {(y_all == 1).sum()}')
print(f'Control (label 0)     : {(y_all == 0).sum()}')

# Hold-out split — stratified by diagnosis
idx = np.arange(len(df_clean))
idx_tr, idx_te = train_test_split(idx, test_size=0.20, stratify=y_all, random_state=RNG)

def take(s, idx_): return s.iloc[idx_].reset_index(drop=True)

X_train, X_test         = take(X_all, idx_tr),    take(X_all, idx_te)
y_train, y_test         = take(y_all, idx_tr),    take(y_all, idx_te)
sites_train, sites_test = take(sites_all, idx_tr),take(sites_all, idx_te)
age_train, age_test     = take(age_all, idx_tr),  take(age_all, idx_te)
sex_train, sex_test     = take(sex_all, idx_tr),  take(sex_all, idx_te)

print(f'\nTrain N = {len(X_train)}   |   Test N = {len(X_test)}')
print(f'Train ASD ratio = {y_train.mean():.3f}   |   Test ASD ratio = {y_test.mean():.3f}')

missing_in_train = set(sites_test) - set(sites_train)
if missing_in_train:
    print(f'\nWARNING: sites only in test (ComBat cannot harmonize): {missing_in_train}')
else:
    print('\nOK: every test site is also represented in training.')


# --- Cell 12 --- 

# Fit on raw (un-harmonized) data — simple holdout
scaler_raw = StandardScaler()
X_tr_raw = scaler_raw.fit_transform(X_train)
X_te_raw = scaler_raw.transform(X_test)

baseline_results = {}
for name, clf in [
    ('SVM (RBF, raw)',  SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RNG)),
    ('CART (raw)',      DecisionTreeClassifier(max_depth=5, random_state=RNG)),
]:
    clf.fit(X_tr_raw, y_train)
    y_pred  = clf.predict(X_te_raw)
    y_proba = clf.predict_proba(X_te_raw)[:, 1]
    baseline_results[name] = compute_metrics(y_test, y_pred, y_proba)

pd.DataFrame(baseline_results).T.round(3)


# --- Cell 14 --- 

# --- 5-fold stratified CV, proper in-fold ComBat ---
print('SVM (RBF) + in-fold ComBat')
print('-' * 60)
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RNG)
svm_cv, svm_preds = cross_validate_pipeline(
    X_train, y_train, sites_train, age_train, sex_train,
    classifier=svm_clf, harmonize=True, verbose=True
)
print()

print('CART + in-fold ComBat')
print('-' * 60)
cart_clf = DecisionTreeClassifier(max_depth=5, random_state=RNG)
cart_cv, cart_preds = cross_validate_pipeline(
    X_train, y_train, sites_train, age_train, sex_train,
    classifier=cart_clf, harmonize=True, verbose=True
)
print()

print('SVM (RBF) + NO harmonization — same folds for comparison')
print('-' * 60)
svm_raw_cv, svm_raw_preds = cross_validate_pipeline(
    X_train, y_train, sites_train, age_train, sex_train,
    classifier=clone(svm_clf), harmonize=False, verbose=True
)


# --- Cell 15 --- 

def summarise(df_cv, name):
    s = df_cv.drop(columns='fold').agg(['mean', 'std']).T
    s.columns = [f'{name}_mean', f'{name}_std']
    return s

cv_summary = pd.concat([
    summarise(svm_cv,     'SVM_combat'),
    summarise(cart_cv,    'CART_combat'),
    summarise(svm_raw_cv, 'SVM_raw'),
], axis=1).round(3)

print('=== 5-fold CV summary (mean ± std across folds) ===')
cv_summary


# --- Cell 16 --- 

metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'auc_roc']
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))

for ax, metric in zip(axes, metrics_to_plot):
    data = pd.DataFrame({
        'SVM + ComBat':  svm_cv[metric].values,
        'CART + ComBat': cart_cv[metric].values,
        'SVM raw':       svm_raw_cv[metric].values,
    })
    sns.boxplot(data=data, ax=ax, palette=['#2ca02c', '#ff7f0e', '#7f7f7f'])
    sns.stripplot(data=data, ax=ax, color='black', size=4, alpha=0.7)
    ax.set_title(metric.replace('_', ' ').upper())
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=15)
    # horizontal reference line at chance / 0.5
    ax.axhline(0.5, ls=':', color='red', alpha=0.5)

plt.suptitle('5-fold CV — Metric distributions across folds', y=1.04, fontweight='bold')
plt.tight_layout()
plt.show()


# --- Cell 18 --- 

# Extract features from df_harm matched to our training subjects
# Match by SUB_ID
if 'SUB_ID' in df_harm.columns:
    df_harm_idx = df_harm.set_index('SUB_ID')
    train_ids = df_clean.iloc[idx_tr]['SUB_ID'].values
    X_train_harm_global = df_harm_idx.loc[train_ids, THICKNESS_COLS].reset_index(drop=True)
else:
    # Fallback: assume df_harm rows align with df_clean
    X_train_harm_global = df_harm.iloc[idx_tr][THICKNESS_COLS].reset_index(drop=True)

print(f'Globally-harmonized training set shape: {X_train_harm_global.shape}')

# Run CV on globally-harmonized data with harmonize=False (because it's already done — leakily)
print('\nSVM CV on globally-harmonized data (LEAKY)')
print('-' * 60)
svm_leaky_cv, _ = cross_validate_pipeline(
    X_train_harm_global, y_train, sites_train, age_train, sex_train,
    classifier=clone(svm_clf), harmonize=False, verbose=True
)

# Compare three regimes
leakage_compare = pd.DataFrame({
    'SVM raw (no ComBat)':       svm_raw_cv.drop(columns='fold').mean(),
    'SVM proper in-fold ComBat': svm_cv.drop(columns='fold').mean(),
    'SVM global ComBat (LEAKY)': svm_leaky_cv.drop(columns='fold').mean(),
}).round(3)
print()
print('=== Comparison: how does global (leaky) ComBat compare to proper in-fold ComBat? ===')
leakage_compare


# --- Cell 19 --- 

# Bar chart of AUC across the three regimes
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = ['SVM raw\n(no ComBat)', 'SVM proper\nin-fold ComBat', 'SVM global\nComBat (LEAKY)']
auc_means = [svm_raw_cv['auc_roc'].mean(), svm_cv['auc_roc'].mean(), svm_leaky_cv['auc_roc'].mean()]
auc_stds  = [svm_raw_cv['auc_roc'].std(),  svm_cv['auc_roc'].std(),  svm_leaky_cv['auc_roc'].std()]
colors    = ['#7f7f7f', '#2ca02c', '#d62728']
bars = ax.bar(labels, auc_means, yerr=auc_stds, capsize=6, color=colors, edgecolor='black')
ax.set_ylabel('AUC-ROC (mean ± std across folds)')
ax.set_title('Three harmonization regimes — same CV folds')
ax.axhline(0.5, ls=':', color='red', alpha=0.6, label='Chance')
ax.set_ylim(0.4, max(auc_means) + 0.1)
for b, v in zip(bars, auc_means):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()


# --- Cell 21 --- 

from itertools import product

def grid_search_with_combat(X, y, sites, age, sex,
                            classifier_class, param_grid, n_splits=3):
    '''Manual grid search with proper in-fold ComBat. Optimises mean CV AUC.'''
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        clf = classifier_class(**params)
        cv_df, _ = cross_validate_pipeline(
            X, y, sites, age, sex,
            classifier=clf, harmonize=True, n_splits=n_splits
        )
        results.append({**params,
                        'auc_mean':  cv_df['auc_roc'].mean(),
                        'auc_std':   cv_df['auc_roc'].std(),
                        'acc_mean':  cv_df['accuracy'].mean(),
                        'sens_mean': cv_df['sensitivity'].mean()})
    return pd.DataFrame(results).sort_values('auc_mean', ascending=False).reset_index(drop=True)


svm_grid = {
    'kernel':       ['rbf', 'linear'],
    'C':            [0.1, 1.0, 10.0],
    'gamma':        ['scale', 'auto'],
    'probability':  [True],
    'random_state': [RNG],
}
print('Grid search — SVM ...')
svm_gs = grid_search_with_combat(X_train, y_train, sites_train, age_train, sex_train,
                                 SVC, svm_grid)
print('Top 10 SVM configurations:')
svm_gs.head(10).round(3)


# --- Cell 22 --- 

cart_grid = {
    'max_depth':         [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
    'random_state':      [RNG],
}
print('Grid search — CART ...')
cart_gs = grid_search_with_combat(X_train, y_train, sites_train, age_train, sex_train,
                                  DecisionTreeClassifier, cart_grid)
print('Top 10 CART configurations:')
cart_gs.head(10).round(3)


# --- Cell 24 --- 

# Cast best params back to native types (grid stores them as object)
def cast_svm(row):
    return {
        'kernel': str(row['kernel']),
        'C': float(row['C']),
        'gamma': str(row['gamma']),
        'probability': bool(row['probability']),
        'random_state': int(row['random_state']),
    }
def cast_cart(row):
    md = row['max_depth']
    md = None if md is None or (isinstance(md, float) and pd.isna(md)) else int(md)
    return {
        'max_depth': md,
        'min_samples_split': int(row['min_samples_split']),
        'criterion': str(row['criterion']),
        'random_state': int(row['random_state']),
    }

best_svm_params  = cast_svm(svm_gs.iloc[0])
best_cart_params = cast_cart(cart_gs.iloc[0])
print('Best SVM params :', best_svm_params)
print('Best CART params:', best_cart_params)

# Fit ComBat on the FULL training set; apply to test
X_train_h, train_estimates = fit_combat_train(X_train, sites_train, age_train, sex_train)
X_test_h = apply_combat_test(X_test, sites_test, train_estimates)

scaler_final = StandardScaler()
X_train_hs = scaler_final.fit_transform(X_train_h)
X_test_hs  = scaler_final.transform(X_test_h)

final_models = {
    'SVM':  SVC(**best_svm_params).fit(X_train_hs, y_train),
    'CART': DecisionTreeClassifier(**best_cart_params).fit(X_train_hs, y_train),
}

final_results, final_proba = {}, {}
for name, mdl in final_models.items():
    y_pred = mdl.predict(X_test_hs)
    y_proba = mdl.predict_proba(X_test_hs)[:, 1]
    final_results[name] = compute_metrics(y_test, y_pred, y_proba)
    final_proba[name]   = y_proba

print('\n=== Final hold-out test results ===')
final_df = pd.DataFrame(final_results).T.round(3)
final_df


# --- Cell 25 --- 

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (name, mdl) in zip(axes, final_models.items()):
    y_pred = mdl.predict(X_test_hs)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Control', 'ASD'], yticklabels=['Control', 'ASD'], cbar=False)
    ax.set_title(f'{name} — confusion matrix')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    ax.text(0.5, -0.27, f'Sensitivity = {sens:.2f}   Specificity = {spec:.2f}\nFN = {fn} cases missed   FP = {fp} false alarms',
            transform=ax.transAxes, ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# Classification reports
for name, mdl in final_models.items():
    print(f'\n=== {name} ===')
    y_pred = mdl.predict(X_test_hs)
    print(classification_report(y_test, y_pred, target_names=['Control', 'ASD']))


# --- Cell 26 --- 

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# ROC
ax = axes[0]
for name, proba in final_proba.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='grey', label='Chance')
ax.set_xlabel('False Positive Rate  (1 − Specificity)')
ax.set_ylabel('True Positive Rate  (Sensitivity)')
ax.set_title('ROC curves — hold-out test set')
ax.legend(loc='lower right')

# Precision-Recall
ax = axes[1]
for name, proba in final_proba.items():
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax.plot(rec, prec, lw=2, label=f'{name} (AP = {ap:.3f})')
ax.axhline(y_test.mean(), ls='--', color='grey', label=f'Baseline (prevalence = {y_test.mean():.2f})')
ax.set_xlabel('Recall  (Sensitivity)')
ax.set_ylabel('Precision  (PPV)')
ax.set_title('Precision-Recall curves')
ax.legend(loc='lower left')

plt.tight_layout()
plt.show()


# --- Cell 28 --- 

# Use the better-performing model by AUC on the hold-out
best_name = max(final_proba, key=lambda k: roc_auc_score(y_test, final_proba[k]))
proba = final_proba[best_name]
print(f'Model used for threshold analysis: {best_name} (AUC = {roc_auc_score(y_test, proba):.3f})')

thresholds = np.linspace(0.01, 0.99, 99)
rows = []
for t in thresholds:
    yp = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    rows.append({
        'threshold': t,
        'sensitivity': sens,
        'specificity': spec,
        'youden_J':    sens + spec - 1,
        'cost_5to1':   5 * fn + 1 * fp,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
    })
thr_df = pd.DataFrame(rows)

t_youden = thr_df.loc[thr_df['youden_J'].idxmax(), 'threshold']
mask85   = thr_df['sensitivity'] >= 0.85
t_sens85 = thr_df.loc[mask85, 'threshold'].max() if mask85.any() else np.nan
t_cost   = thr_df.loc[thr_df['cost_5to1'].idxmin(), 'threshold']

def report_thr(t, label):
    if np.isnan(t):
        print(f'  {label:30s}: no threshold satisfies the constraint'); return
    r = thr_df.iloc[(thr_df['threshold'] - t).abs().argmin()]
    print(f'  {label:30s}: t = {t:.2f}   sens = {r["sensitivity"]:.3f}   '
          f'spec = {r["specificity"]:.3f}   FN = {int(r["FN"])}   FP = {int(r["FP"])}')

print(f'\nOptimal thresholds for {best_name}:')
report_thr(0.50,      'Default (0.50)')
report_thr(t_youden,  "Youden's J max")
report_thr(t_sens85,  'Sensitivity >= 0.85')
report_thr(t_cost,    'Minimize 5·FN + 1·FP')


# --- Cell 29 --- 

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

# Left: sens/spec/J vs threshold
ax = axes[0]
ax.plot(thr_df['threshold'], thr_df['sensitivity'], label='Sensitivity (ASD recall)',
        color='#d62728', lw=2)
ax.plot(thr_df['threshold'], thr_df['specificity'], label='Specificity',
        color='#1f77b4', lw=2)
ax.plot(thr_df['threshold'], thr_df['youden_J'],   label="Youden's J",
        color='#2ca02c', lw=2, ls='--')
ax.axvline(0.5,       color='grey',    ls=':', label='Default 0.5')
ax.axvline(t_youden,  color='#2ca02c', ls=':', alpha=0.7, label=f'J-opt {t_youden:.2f}')
if not np.isnan(t_sens85):
    ax.axvline(t_sens85, color='orange', ls=':', label=f'Sens>=0.85 @ {t_sens85:.2f}')
ax.set_xlabel('Decision threshold')
ax.set_ylabel('Value')
ax.set_title(f'Sensitivity / Specificity vs threshold — {best_name}')
ax.legend(loc='center left')

# Right: cost
ax = axes[1]
ax.plot(thr_df['threshold'], thr_df['cost_5to1'], color='black', lw=2)
ax.axvline(0.5,    color='grey', ls=':', label='Default 0.5')
ax.axvline(t_cost, color='red',  ls='--', label=f'Cost-min t = {t_cost:.2f}')
ax.set_xlabel('Decision threshold')
ax.set_ylabel('Total cost   (5 · FN + 1 · FP)')
ax.set_title('Asymmetric clinical cost   (FN penalty = 5 × FP penalty)')
ax.legend()

plt.tight_layout()
plt.show()


# --- Cell 31 --- 

cart_final = final_models['CART']
fi = pd.DataFrame({
    'region':     X_train.columns,
    'importance': cart_final.feature_importances_,
}).sort_values('importance', ascending=False).reset_index(drop=True)
top_fi = fi.head(15)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax = axes[0]
sns.barplot(data=top_fi, y='region', x='importance', ax=ax, color='#ff7f0e')
ax.set_title('CART — Top 15 most important regions')
ax.set_xlabel('Gini importance')

ax = axes[1]
plot_tree(cart_final,
          feature_names=list(X_train.columns),
          class_names=['Control', 'ASD'],
          filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
ax.set_title('CART decision tree (first 3 levels shown)')

plt.tight_layout()
plt.show()

print('\nTop 10 regions by CART importance:')
print(fi.head(10).round(4).to_string(index=False))


# --- Cell 32 --- 

# Refit a linear-kernel SVM purely for interpretation
lin_svm = SVC(kernel='linear', C=1.0, probability=True, random_state=RNG)
lin_svm.fit(X_train_hs, y_train)

coefs = pd.DataFrame({
    'region':   X_train.columns,
    'coef':     lin_svm.coef_[0],
    'abs_coef': np.abs(lin_svm.coef_[0]),
}).sort_values('abs_coef', ascending=False).reset_index(drop=True)
top_coefs = coefs.head(15).copy()
colors = ['#d62728' if v < 0 else '#1f77b4' for v in top_coefs['coef']]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_coefs['region'][::-1], top_coefs['coef'][::-1], color=colors[::-1])
ax.set_xlabel('Linear SVM coefficient   (sign = direction, magnitude = importance)')
ax.set_title('Top 15 regions by |linear SVM coefficient|\n'
             '(blue: pushes toward ASD ; red: pushes toward Control)')
ax.axvline(0, color='black', lw=0.8)
plt.tight_layout()
plt.show()

# Quick sanity: linear SVM hold-out performance
y_pred_lin  = lin_svm.predict(X_test_hs)
y_proba_lin = lin_svm.predict_proba(X_test_hs)[:, 1]
print('Linear SVM on hold-out test:')
print(pd.Series(compute_metrics(y_test, y_pred_lin, y_proba_lin)).round(3))


# --- Cell 33 --- 

# 8 FDR-confirmed regions from the RQ1 report
rq1_confirmed = [
    'lh_lateraloccipital', 'rh_lateraloccipital',
    'lh_cuneus',           'lh_entorhinal',
    'lh_lingual',          'rh_lingual',
    'lh_medialorbitofrontal', 'lh_pericalcarine',
]
rq1_set  = set(rq1_confirmed)
top_cart = set(fi.head(15)['region'].tolist())
top_svm  = set(coefs.head(15)['region'].tolist())

print('=' * 70)
print('Overlap between classifier top-15 regions and the 8 RQ1 FDR-confirmed regions')
print('=' * 70)
print(f'\nRQ1 confirmed (8 regions):  {sorted(rq1_set)}')
print(f'\nCART top-15 ∩ RQ1 confirmed: {sorted(top_cart & rq1_set)}')
print(f'SVM  top-15 ∩ RQ1 confirmed: {sorted(top_svm  & rq1_set)}')
print(f'\nCART caught {len(top_cart & rq1_set)} / 8 of the RQ1 regions in its top 15.')
print(f'SVM  caught {len(top_svm  & rq1_set)} / 8 of the RQ1 regions in its top 15.')

# Visualize overlap as a Venn-like bar chart
fig, ax = plt.subplots(figsize=(9, 4.5))
all_regions = sorted(rq1_set | top_cart | top_svm)
data = pd.DataFrame({
    'region':  all_regions,
    'RQ1':     [r in rq1_set  for r in all_regions],
    'CART15':  [r in top_cart for r in all_regions],
    'SVM15':   [r in top_svm  for r in all_regions],
}).set_index('region')
# Restrict to RQ1 regions for the plot
data_rq1 = data.loc[sorted(rq1_set)].astype(int)
data_rq1.plot(kind='barh', stacked=False, ax=ax,
              color=['#9467bd', '#ff7f0e', '#1f77b4'])
ax.set_title('Which RQ1-confirmed regions does each classifier pick up?')
ax.set_xlabel('Presence (1 = in that set)')
ax.set_xlim(0, 1.4)
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

