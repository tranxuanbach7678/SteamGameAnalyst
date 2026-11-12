"""Language parsing and reproducible contrasts for the Q2/Q4 notebook analyses."""

import ast
import html
import re

import numpy as np
import pandas as pd


def parse_language_count(value, empty_list_is_zero=False):
    """Count unique language labels; distinguish unknown metadata from no audio.

    Supported languages=[] is treated as unknown, while Full audio languages=[]
    means no recorded full audio. Missing/blank values remain unknown in both.
    """
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return np.nan
    if isinstance(value, (list, tuple)):
        labels = value
    else:
        text = str(value).strip()
        if not text or text.casefold() in {'nan', 'none', 'null'}:
            return np.nan
        try:
            labels = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Handle unquoted comma-separated storefront exports as well.
            if text.startswith(('[', '{', '(')):
                return np.nan
            labels = text.split(',')
        if not isinstance(labels, (list, tuple)):
            return np.nan
    labels = {
        re.sub(r'<[^>]+>|\*', '', html.unescape(str(label))).strip().casefold()
        for label in labels
    } - {''}
    return len(labels) if labels or empty_list_is_zero else np.nan


def bootstrap_contrast(data, group_col, outcome, strata=(), statistic='mean',
                       min_n=5, n_resamples=1000, seed=42):
    """Estimate group True minus False on common strata with fixed pooled weights.

    Each stratum must contain min_n observed games per group. Weights are the
    pooled game shares among eligible strata, so both groups use the same mix.
    Bootstrap resamples games independently within each group/stratum. Intervals
    are conditional on the observed strata and assume independent games.
    """
    if statistic not in {'mean', 'median'}:
        raise ValueError('statistic must be mean or median')
    reducer = np.mean if statistic == 'mean' else np.median
    cols = list(dict.fromkeys([group_col, outcome, *strata]))
    work = data[cols].replace([np.inf, -np.inf], np.nan).dropna()
    groups = work.groupby(list(strata), observed=True, sort=True) if strata else [(None, work)]
    samples = []
    for _, part in groups:
        treated = part.loc[part[group_col].eq(True), outcome].to_numpy(dtype=float)
        control = part.loc[part[group_col].eq(False), outcome].to_numpy(dtype=float)
        if min(len(treated), len(control)) >= min_n:
            samples.append((treated, control))
    n1 = sum(len(a) for a, _ in samples)
    n0 = sum(len(b) for _, b in samples)
    result = dict(N_Treated=n1, N_Control=n0, Strata=len(samples),
                  Coverage=(n1 + n0) / len(work) if len(work) else 0.0,
                  Estimate=np.nan, CI_Low=np.nan, CI_High=np.nan)
    if not samples:
        return result
    rng = np.random.default_rng(seed)
    draws = np.zeros(n_resamples)
    estimate = 0.0
    for treated, control in samples:
        weight = (len(treated) + len(control)) / (n1 + n0)
        estimate += weight * (reducer(treated) - reducer(control))
        # Batching bounds memory even for large Steam cohorts.
        for start in range(0, n_resamples, 50):
            stop = min(start + 50, n_resamples)
            a = treated[rng.integers(len(treated), size=(stop - start, len(treated)))]
            b = control[rng.integers(len(control), size=(stop - start, len(control)))]
            draws[start:stop] += weight * (reducer(a, axis=1) - reducer(b, axis=1))
    result['Estimate'] = estimate
    if n_resamples:
        result['CI_Low'], result['CI_High'] = np.percentile(draws, [2.5, 97.5])
    return result
