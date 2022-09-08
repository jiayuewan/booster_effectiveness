import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def create_regression_formula(features):
    """Generates a regression formula with specified covariates."""
    formula = "infection ~ 1"
    for feature in features:
        if 'vac_type' in feature:
            formula = ' + '.join([formula, f"C({feature}, Treatment('Pfizer'))"])
        elif feature == 'student_group':
            formula = ' + '.join([formula, f"C({feature}, Treatment('UG-other'))"])
        elif feature == 'last_dose_month':
            formula = ' + '.join([formula, f"C({feature}, Treatment(5))"])
        elif feature == 'week':
            formula = ' + '.join([formula, f"C({feature}, Treatment(0))"])
        elif feature == 'building':
            formula = ' + '.join([formula, f"C({feature}, Treatment('Ganedago: Hall'))"])
        else:
            formula = ' + '.join([formula, f"C({feature})"])
    print(formula)
    return formula


def run_logistic_regression(features, df, verbose=True):
    """Run logistic regression.

    Args:
        features: potential confounding variables for the regression.
        df: A pandas dataframe used for regression.

    Returns:
        Regression results.
    """
    formula = create_regression_formula(features)
    model = smf.logit(formula=formula, data=df)
    res = model.fit(maxiter=1000, method='lbfgs')
    if verbose:
        print(res.summary())
    return res.params[1], res.conf_int().iloc[1], res


def run_gee_poisson(features, df, verbose=True):
    """Run Poisson regression with generalized estimating equations (GEE)."""
    formula = create_regression_formula(features)
    df_pois = df.groupby(["employee_id_hash"] + features, as_index=False).agg({'day': 'count', 'infection': 'max'})
    fam = sm.families.Poisson()
    cov_struct = sm.cov_struct.Exchangeable()
    model = smf.gee(
        formula=formula,
        data=df_pois,
        groups="employee_id_hash",
        time="week",
        cov_struct=cov_struct,
        family=fam,
        offset=np.log(df_pois['day']),
        update_dep=True,
    )
    res = model.fit(maxiter=300)
    if verbose:
        print(res.summary())
    return res.params[1], res.conf_int().iloc[1], res.cov_struct.dep_params, res


def gen_result_summary(res):
    """Compute summarized regression results."""
    n_params = res.params.shape[0]
    summary = res.params.to_frame(name='coeff')
    summary['SE'] = res.bse
    adjusted_p_values = multipletests(res.pvalues, alpha=0.05, method='bonferroni')[1] # bonferroni correction
    summary['p-value'] = res.pvalues
    summary['adj p-value'] = adjusted_p_values
    summary['aOR'] = summary.coeff.apply(np.exp)

    alpha = 0.05
    alpha_corrected = 0.05 / n_params # bonferroni correction
    z_score = st.norm.ppf(1 - alpha_corrected / 2)
    summary['aOR lower'] = np.exp(summary.coeff - summary.SE * z_score)
    summary['aOR upper'] = np.exp(summary.coeff + summary.SE * z_score)
    return summary


def run_delay_sensitivity_analysis(regression_runner, features):
    """Perform sensitivity analysis on delay for the booster to become effectiveness.

    Args:
        regression_runner: A callable running a regression.
        features: A list of covariates to be included in the regression.

    Returns:
        A 2-d numpy array where row `i` is the mean, lb and ub for the booster
        effectiveness, assuming the delay for the booster to become effective
        is `i+1` days.
    """
    ans = np.zeros((14, 3))
    delays =  list(range(1, 15))
    for i, d in enumerate(delays):
        print(f'delay = {d}\n')
        df = pd.read_csv(f"../data/regression/data_indiv_day_delay_{d}.csv")
        param, conf_int, corr, res = regression_runner(features, df, verbose=False)
        summary = gen_result_summary(res)
        ans[i,:] = summary.iloc[1, -3:]
    booster_effectiveness = 1 - ans
    return booster_effectiveness
