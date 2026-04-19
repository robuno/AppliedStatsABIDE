"""
All functions except covbat are forked from
https://github.com/brentp/combat.py
combat function modified to enable correction without empirical Bayes
covbat function written by Andrew Chen (andrewac@pennmedicine.upenn.edu)
"""
import pandas as pd
import patsy
import sys
import numpy.linalg as la
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def adjust_nums(numerical_covariates, drop_idxs):
    # if we dropped some values, have to adjust those with a larger index.
    if numerical_covariates is None: return drop_idxs
    return [nc - sum(nc < di for di in drop_idxs) for nc in numerical_covariates]

def design_mat(mod, numerical_covariates, batch_levels):
    # require levels to make sure they are in the same order as we use in the
    # rest of the script.
    design = patsy.dmatrix("~ 0 + C(batch, levels=%s)" % str(batch_levels),
                                                  mod, return_type="dataframe")

    mod = mod.drop(["batch"], axis=1)
    numerical_covariates = list(numerical_covariates)
    sys.stderr.write("found %i batches\n" % design.shape[1])
    other_cols = [c for i, c in enumerate(mod.columns)
                  if not i in numerical_covariates]
    factor_matrix = mod[other_cols]
    design = pd.concat((design, factor_matrix), axis=1)
    if numerical_covariates is not None:
        sys.stderr.write("found %i numerical covariates...\n"
                            % len(numerical_covariates))
        for i, nC in enumerate(numerical_covariates):
            cname = mod.columns[nC]
            sys.stderr.write("\t{0}\n".format(cname))
            design[cname] = mod[mod.columns[nC]]
    sys.stderr.write("found %i categorical variables:" % len(other_cols))
    sys.stderr.write("\t" + ", ".join(other_cols) + '\n')
    return design

"""Correction of *Cov*ariance *Bat* effects

Parameters
----------
data : pandas.DataFrame
    A (n_features, n_samples) dataframe of the expression or methylation
    data to batch correct
batch : pandas.Series
    A column corresponding to the batches in the data, with index same as
    the columns that appear in ``data``
model : patsy.design_info.DesignMatrix, optional
    A model matrix describing metadata on the samples which could be
    causing batch effects. If not provided, then will attempt to coarsely
    correct just from the information provided in ``batch``
numerical_covariates : list-like
    List of covariates in the model which are numerical, rather than
    categorical
pct_var : numeric
    Numeric between 0 and 1 indicating the percent of variation that is
    explained by the adjusted PCs
n_pc : numeric
    If >0, then this specifies the number of PCs to adjust. Overrides pct_var

Returns
-------
corrected : pandas.DataFrame
    A (n_features, n_samples) dataframe of the batch-corrected data
"""
def covbat(data, batch, model=None, numerical_covariates=None, pct_var=0.95, n_pc=0):
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({'batch': batch})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in  ((model == 1).all()).items() if inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if not c in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
        for c in numerical_covariates if not c in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)

    sys.stderr.write("Standardizing Data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((int(n_array), 1)) / int(n_array))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array)))))

    sys.stderr.write("Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []

    for i, batch_idxs in enumerate(batch_info):
        #batches = [list(model.columns).index(b) for b in batches]
        delta_hat.append(s_data[batch_idxs].var(axis=1))

    gamma_bar = gamma_hat.mean(axis=1) 
    t2 = gamma_hat.var(axis=1)
   
    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        #print '18 20 22 28 29 31 32 33 35 40 46'
        #print batch_info[batch_id]

        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                     delta_hat[i], gamma_bar[i], t2[i], a_prior[i], b_prior[i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    sys.stdout.write("Adjusting data\n")
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    for j, batch_idxs in enumerate(batch_info):

        dsq = np.sqrt(delta_star[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom =  np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)

        bayesdata[batch_idxs] = numer / denom
   
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    # not adding back stand_mean yet
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array))))

    # CovBat step: PCA then ComBat without EB on the scores
    # comdata = data.T
    comdata = bayesdata.T
    bmu = np.mean(comdata, axis=0)
    # standardize data before PCA
    scaler = StandardScaler()
    comdata = scaler.fit_transform(comdata)
    
    pca = PCA()
    pca.fit(comdata)
    pc_comp = pca.components_
    full_scores = pd.DataFrame(pca.fit_transform(comdata)).T
    full_scores.columns = data.columns

    var_exp=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
    npc = np.min(np.where(var_exp>pct_var))+1
    if n_pc > 0:
        npc = n_pc
    scores = full_scores.loc[range(0,npc),:]
    scores_com = combat(scores, batch, model=None, eb=False)
    full_scores.loc[range(0,npc),:] = scores_com

    x_covbat = bayesdata - bayesdata # create pandas DataFrame to store output
    # x_covbat = x_covbat.add(bmu, axis='index')
    proj = np.dot(full_scores.T, pc_comp).T
    x_covbat += scaler.inverse_transform(proj.T).T
    # x_covbat = x_covbat * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean
    x_covbat += stand_mean
 
    return x_covbat

def combat(data, batch, model=None, numerical_covariates=None, eb=True):
    """Correct for batch effects in a dataset

    Parameters
    ----------
    data : pandas.DataFrame
        A (n_features, n_samples) dataframe of the expression or methylation
        data to batch correct
    batch : pandas.Series
        A column corresponding to the batches in the data, with index same as
        the columns that appear in ``data``
    model : patsy.design_info.DesignMatrix, optional
        A model matrix describing metadata on the samples which could be
        causing batch effects. If not provided, then will attempt to coarsely
        correct just from the information provided in ``batch``
    numerical_covariates : list-like
        List of covariates in the model which are numerical, rather than
        categorical
    eb : logical
        Should empirical Bayes adjustments be made, if FALSE then gamma_hat
        and delta_hat are used as correction

    Returns
    -------
    corrected : pandas.DataFrame
        A (n_features, n_samples) dataframe of the batch-corrected data
    """
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({'batch': batch})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in  ((model == 1).all()).items() if inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if not c in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
            for c in numerical_covariates if not c in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)

    sys.stderr.write("Standardizing Data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((int(n_array), 1)) / int(n_array))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array)))))

    sys.stderr.write("Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []

    for i, batch_idxs in enumerate(batch_info):
        #batches = [list(model.columns).index(b) for b in batches]
        delta_hat.append(s_data[batch_idxs].var(axis=1))

    gamma_bar = gamma_hat.mean(axis=1) 
    t2 = gamma_hat.var(axis=1)
   

    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        #print '18 20 22 28 29 31 32 33 35 40 46'
        #print batch_info[batch_id]

        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                     delta_hat[i], gamma_bar[i], t2[i], a_prior[i], b_prior[i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    sys.stdout.write("Adjusting data\n")
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)


    for j, batch_idxs in enumerate(batch_info):
        if eb:
            dsq = np.sqrt(delta_star[j,:])
            dsq = dsq.reshape((len(dsq), 1))
            denom =  np.dot(dsq, np.ones((1, n_batches[j])))
            numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)

            bayesdata[batch_idxs] = numer / denom
        else:
            gamma_hat = np.array(gamma_hat)
            delta_hat = np.array(delta_hat)
            
            dsq = np.sqrt(delta_hat[j,:])
            dsq = dsq.reshape((len(dsq), 1))
            denom =  np.dot(dsq, np.ones((1, n_batches[j])))
            numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_hat).T)

            bayesdata[batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean
 
    return bayesdata

def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        #print g_hat.shape, g_bar.shape, t2.shape
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.values.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)
       
        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 

    

def aprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (2 * s2 +m**2) / s2

def bprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


# =====================================================================
# Fit / Transform API for CovBat
# =====================================================================

def covbat_fit(data, batch, model=None, numerical_covariates=None,
               pct_var=0.95, n_pc=0):
    """Learn CovBat harmonization parameters from training data.

    Follows the same logic as covbat() but stores all learned
    parameters in a dictionary instead of returning corrected data.
    Use covbat_transform() to apply these parameters to new data.

    Parameters
    ----------
    data : pd.DataFrame, shape (n_features, n_samples)
    batch : pd.Series with sample batch/site labels
    model : pd.DataFrame of biological covariates to protect
    numerical_covariates : list of numerical covariate column names
    pct_var : float, cumulative variance threshold for PCs
    n_pc : int, override number of PCs (0 = use pct_var)

    Returns
    -------
    covbat_params : dict  – all parameters needed by covbat_transform()
    """
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model = model.copy()
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({'batch': batch})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in ((model == 1).all()).items()
                 if inter == True]
    model = model[[c for c in model.columns if c not in drop_cols]]
    numerical_covariates = [
        list(model.columns).index(c) if isinstance(c, str) else c
        for c in numerical_covariates if c not in drop_cols
    ]

    design = design_mat(model, numerical_covariates, batch_levels)

    # ── Step 1: ComBat standardization parameters ────────────────────
    sys.stderr.write("[covbat_fit] Standardizing data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T),
                   data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch, :])
    var_pooled = np.dot(
        ((data - np.dot(design, B_hat).T) ** 2),
        np.ones((int(n_array), 1)) / int(n_array),
    )

    stand_mean = np.dot(
        grand_mean.T.reshape((len(grand_mean), 1)),
        np.ones((1, int(n_array))),
    )
    tmp = np.array(design.copy())
    tmp[:, :n_batch] = 0
    stand_mean += np.dot(tmp, B_hat).T

    s_data = (data - stand_mean) / np.dot(
        np.sqrt(var_pooled), np.ones((1, int(n_array)))
    )

    sys.stderr.write("[covbat_fit] Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(
        np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T),
        s_data.T,
    )

    delta_hat = []
    for i, batch_idxs in enumerate(batch_info):
        delta_hat.append(s_data[batch_idxs].var(axis=1))

    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1)

    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("[covbat_fit] Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                      delta_hat[i], gamma_bar[i], t2[i],
                      a_prior[i], b_prior[i])
        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    # ── Step 2: Apply ComBat correction → bayesdata ──────────────────
    sys.stderr.write("[covbat_fit] Adjusting data\n")
    bayesdata = s_data.copy()
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    for j, batch_idxs in enumerate(batch_info):
        dsq = np.sqrt(delta_star[j, :]).reshape((-1, 1))
        denom = np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(
            bayesdata[batch_idxs]
            - np.dot(batch_design.loc[batch_idxs], gamma_star).T
        )
        bayesdata[batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    # vpsq-scaled result WITHOUT stand_mean
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array))))

    # ── Step 3: PCA parameters ───────────────────────────────────────
    comdata = bayesdata.T
    scaler = StandardScaler()
    comdata_scaled = scaler.fit_transform(comdata)

    pca = PCA()
    pca.fit(comdata_scaled)
    pc_comp = pca.components_
    full_scores = pd.DataFrame(pca.transform(comdata_scaled)).T
    full_scores.columns = data.columns

    var_exp = np.cumsum(
        np.round(pca.explained_variance_ratio_, decimals=4)
    )
    npc = int(np.min(np.where(var_exp > pct_var)) + 1)
    if n_pc > 0:
        npc = n_pc

    # ── Step 4: ComBat on PC scores (eb=False) — learn parameters ────
    scores = full_scores.loc[range(0, npc), :]

    # Replicate the combat() internals for eb=False to store params
    scores_model = pd.DataFrame({'batch': batch})
    scores_batch_items = scores_model.groupby("batch").groups.items()
    scores_batch_levels = [k for k, v in scores_batch_items]
    scores_batch_info = [v for k, v in scores_batch_items]
    scores_n_batch = len(scores_batch_info)
    scores_n_batches = np.array([len(v) for v in scores_batch_info])
    scores_n_array = float(sum(scores_n_batches))

    scores_design = design_mat(
        scores_model.copy(), [], scores_batch_levels
    )
    scores_B_hat = np.dot(
        np.dot(la.inv(np.dot(scores_design.T, scores_design)),
               scores_design.T),
        scores.T,
    )
    scores_grand_mean = np.dot(
        (scores_n_batches / scores_n_array).T,
        scores_B_hat[:scores_n_batch, :],
    )
    scores_var_pooled = np.dot(
        ((scores - np.dot(scores_design, scores_B_hat).T) ** 2),
        np.ones((int(scores_n_array), 1)) / int(scores_n_array),
    )
    scores_stand_mean = np.dot(
        scores_grand_mean.T.reshape((len(scores_grand_mean), 1)),
        np.ones((1, int(scores_n_array))),
    )

    scores_s_data = (scores - scores_stand_mean) / np.dot(
        np.sqrt(scores_var_pooled),
        np.ones((1, int(scores_n_array))),
    )

    scores_batch_design = scores_design[
        scores_design.columns[:scores_n_batch]
    ]
    scores_gamma_hat = np.dot(
        np.dot(
            la.inv(np.dot(scores_batch_design.T, scores_batch_design)),
            scores_batch_design.T,
        ),
        scores_s_data.T,
    )
    scores_delta_hat = []
    for i, bidxs in enumerate(scores_batch_info):
        scores_delta_hat.append(scores_s_data[bidxs].var(axis=1))

    # ── Build the parameter dictionary ───────────────────────────────
    covbat_params = {
        # ComBat level-1 parameters
        'grand_mean': grand_mean,
        'var_pooled': var_pooled,
        'stand_mean': stand_mean,
        'gamma_star': gamma_star,
        'delta_star': delta_star,
        'batch_levels': batch_levels,
        'batch_info': batch_info,
        'n_batch': n_batch,
        'n_batches': n_batches,
        'design_columns': design.columns.tolist(),
        'B_hat': B_hat,
        # PCA parameters
        'scaler': scaler,
        'pca': pca,
        'pc_comp': pc_comp,
        'npc': npc,
        # Score-level ComBat parameters (eb=False)
        'scores_gamma_hat': np.array(scores_gamma_hat),
        'scores_delta_hat': [np.array(d) for d in scores_delta_hat],
        'scores_stand_mean': scores_stand_mean,
        'scores_var_pooled': scores_var_pooled,
        'scores_grand_mean': scores_grand_mean,
        'scores_batch_levels': scores_batch_levels,
    }

    sys.stderr.write("[covbat_fit] Done — parameters stored.\n")
    return covbat_params


def covbat_transform(data, batch, covbat_params):
    """Apply stored CovBat parameters to new data without refitting.

    Parameters
    ----------
    data : pd.DataFrame, shape (n_features, n_samples)
        New data to harmonize (e.g. test set).
    batch : pd.Series with batch/site labels for new samples.
    covbat_params : dict returned by covbat_fit().

    Returns
    -------
    x_covbat : pd.DataFrame, shape (n_features, n_samples)
        Harmonized data.
    """
    # Unpack stored parameters
    grand_mean  = covbat_params['grand_mean']
    var_pooled  = covbat_params['var_pooled']
    gamma_star  = covbat_params['gamma_star']
    delta_star  = covbat_params['delta_star']
    batch_levels = covbat_params['batch_levels']
    stand_mean_train = covbat_params['stand_mean']
    B_hat       = covbat_params['B_hat']
    n_batch     = covbat_params['n_batch']
    scaler      = covbat_params['scaler']
    pca         = covbat_params['pca']
    pc_comp     = covbat_params['pc_comp']
    npc         = covbat_params['npc']

    n_samples = data.shape[1]

    # ── Step 1: Apply ComBat correction using stored parameters ──────
    # Build stand_mean for the new data using grand_mean only
    # (no covariate adjustment — covariates were only used for fitting)
    stand_mean_new = np.dot(
        grand_mean.T.reshape((len(grand_mean), 1)),
        np.ones((1, n_samples)),
    )

    s_data = (data - stand_mean_new) / np.dot(
        np.sqrt(var_pooled), np.ones((1, n_samples))
    )

    # Map new samples to their batch index in the training batch_levels
    batch_arr = np.array(batch)
    batch_level_to_idx = {lvl: idx for idx, lvl in enumerate(batch_levels)}

    bayesdata = s_data.copy()
    for j, lvl in enumerate(batch_levels):
        col_mask = batch_arr == lvl
        n_in_batch = col_mask.sum()
        if n_in_batch == 0:
            continue

        sample_indices = data.columns[col_mask]

        dsq = np.sqrt(delta_star[j, :]).reshape((-1, 1))
        denom = np.dot(dsq, np.ones((1, n_in_batch)))

        # gamma_star[j] is the batch effect for this batch
        gamma_j = gamma_star[j, :].reshape((-1, 1))
        numer = np.array(bayesdata[sample_indices]) - gamma_j

        bayesdata[sample_indices] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, n_samples)))

    # ── Step 2: Scale using stored scaler ────────────────────────────
    comdata = scaler.transform(bayesdata.T)

    # ── Step 3: Project onto stored PCA ──────────────────────────────
    full_scores = pd.DataFrame(pca.transform(comdata)).T
    full_scores.columns = data.columns

    # ── Step 4: Apply stored score-level ComBat parameters ───────────
    scores_gamma_hat  = covbat_params['scores_gamma_hat']
    scores_delta_hat  = covbat_params['scores_delta_hat']
    scores_stand_mean = covbat_params['scores_stand_mean']
    scores_var_pooled = covbat_params['scores_var_pooled']
    scores_grand_mean = covbat_params['scores_grand_mean']
    scores_batch_levels = covbat_params['scores_batch_levels']

    scores = full_scores.loc[range(0, npc), :].copy()

    # Build stand_mean for scores on new data (grand_mean only)
    sc_stand_mean_new = np.dot(
        scores_grand_mean.T.reshape((len(scores_grand_mean), 1)),
        np.ones((1, n_samples)),
    )

    sc_s_data = (scores - sc_stand_mean_new) / np.dot(
        np.sqrt(scores_var_pooled), np.ones((1, n_samples))
    )

    # Apply stored gamma_hat / delta_hat (eb=False) per batch
    sc_bayesdata = sc_s_data.copy()
    for j, lvl in enumerate(scores_batch_levels):
        col_mask = batch_arr == lvl
        n_in_batch = col_mask.sum()
        if n_in_batch == 0:
            continue

        sample_indices = data.columns[col_mask]

        sc_delta_j = np.array(scores_delta_hat[j])
        dsq = np.sqrt(sc_delta_j).reshape((-1, 1))
        denom = np.dot(dsq, np.ones((1, n_in_batch)))

        gamma_j = scores_gamma_hat[j, :].reshape((-1, 1))
        numer = np.array(sc_bayesdata[sample_indices]) - gamma_j

        sc_bayesdata[sample_indices] = numer / denom

    # Rescale scores back
    sc_vpsq = np.sqrt(scores_var_pooled).reshape(
        (len(scores_var_pooled), 1)
    )
    sc_corrected = sc_bayesdata * np.dot(
        sc_vpsq, np.ones((1, n_samples))
    ) + sc_stand_mean_new

    full_scores.loc[range(0, npc), :] = sc_corrected

    # ── Step 5: Inverse transform back to feature space ──────────────
    proj = np.dot(full_scores.T, pc_comp).T
    x_covbat = pd.DataFrame(
        scaler.inverse_transform(proj.T).T,
        index=data.index,
        columns=data.columns,
    )
    x_covbat += stand_mean_new

    sys.stderr.write("[covbat_transform] Done.\n")
    return x_covbat
