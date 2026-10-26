"""E5: fixed-origin temporal backtests and aligned residual analysis.

Run from repository root: .venv/bin/python -m experiments.error_analysis.e5
No test.csv, E4, tuning, target calibration or changes to official E2/E3 outputs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.evaluation import evaluate
from src.features import make_listing
from src.validation import load_data
from experiments.feature_engineering.utils.core import ROOT, CORE, F0, PARAMS, encode, composite
from experiments.feature_engineering.utils.auxiliary import build, prepare_auxiliary, GROUP_FEATURES

DEST = Path(__file__).resolve().parent
CONFIGS = {
    'F0': None,
    'Core': (),
    'HDB': ('hdb',),
    'HDB_MRT': ('hdb', 'mrt'),
    'Schools': ('hdb', 'mrt', 'schools'),
    'Malls': ('hdb', 'mrt', 'schools', 'malls'),
    'COE': ('hdb', 'mrt', 'schools', 'coe'),
    'Stock': ('hdb', 'mrt', 'schools', 'stock'),
    'All': ('hdb', 'mrt', 'schools', 'malls', 'coe', 'stock'),
}
# Freeze before execution; both validation windows end inside original training.
FOLDS = [('origin_2022_06', '2022-06', 16), ('origin_2022_12', '2022-12', 16)]
CONTROLS = {'Core': 'F0', 'HDB': 'Core', 'HDB_MRT': 'HDB', 'Schools': 'HDB_MRT',
            'Malls': 'Schools', 'COE': 'Schools', 'Stock': 'Schools', 'All': 'Schools'}


def split_at(frame, cutoff, horizon):
    months = pd.PeriodIndex(frame.RENT_APPROVAL_DATE, freq='M')
    origin = pd.Period(cutoff, freq='M')
    train = frame.loc[months <= origin].copy()
    valid = frame.loc[(months > origin) & (months <= origin + horizon)].copy()
    if train.empty or valid.empty or train.index.intersection(valid.index).size:
        raise ValueError('Empty or overlapping split.')
    observed = set(pd.PeriodIndex(valid.RENT_APPROVAL_DATE, freq='M'))
    if observed != set(pd.period_range(origin + 1, periods=horizon, freq='M')):
        raise ValueError('Incomplete forecast horizon.')
    return train, valid


def validate_predictions(predictions, valid):
    if not predictions.index.equals(valid.index) or not predictions.index.is_unique:
        raise ValueError('Prediction index/order does not match validation rows.')
    np.testing.assert_array_equal(predictions.actual, valid.MONTHLY_RENT)
    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError('Nonfinite predictions.')


def profile(train, valid, cutoff):
    """Group edges and address support use training rows only."""
    p = pd.DataFrame(index=valid.index)
    p['month'] = valid.RENT_APPROVAL_DATE
    p['horizon'] = [m.ordinal - pd.Period(cutoff, freq='M').ordinal
                    for m in pd.PeriodIndex(valid.RENT_APPROVAL_DATE, freq='M')]
    p['horizon_band'] = pd.cut(p.horizon, [0, 4, 8, 12, 16],
                               labels=['01-04', '05-08', '09-12', '13-16']).astype(str)
    for col in ['TOWN', 'FLAT_TYPE']:
        p[col] = valid[col].fillna('missing')
    p['area_band'] = pd.cut(valid.FLOOR_AREA_SQM, [-np.inf, 60, 90, 110, np.inf],
                             labels=['<=60', '60-90', '90-110', '>110']).astype(str)
    counts = composite(make_listing(train)).value_counts()
    support = composite(make_listing(valid)).map(counts).fillna(0)
    p['address_support'] = pd.cut(support, [-1, 0, 5, 20, np.inf],
                                  labels=['unseen', '1-5', '6-20', '>20']).astype(str)
    edges = np.unique(np.r_[-np.inf, train.MONTHLY_RENT.quantile([.25, .5, .75, .9]).values, np.inf])
    # True-price groups are post-hoc diagnostics, never available prediction inputs.
    p['actual_rent_band'] = pd.cut(valid.MONTHLY_RENT, edges).astype(str)
    return p, edges.tolist()


def summarize(fold, train, valid, cutoff, pred):
    validate_predictions(pred, valid)
    p, edges = profile(train, valid, cutoff)
    overall, groups, oracles = [], [], []
    dimensions = ['month', 'horizon_band', 'TOWN', 'FLAT_TYPE', 'area_band',
                  'address_support', 'actual_rent_band']
    for name in CONFIGS:
        error = pred[name] - pred.actual
        sq = error ** 2
        row = dict(fold=fold, configuration=name, train_rows=len(train), validation_rows=len(valid),
                   cutoff=cutoff, validation_end=valid.RENT_APPROVAL_DATE.max(),
                   **evaluate(pred.actual, pred[name]))
        overall.append(row)
        for dimension in dimensions:
            for label, idx in p.groupby(dimension, observed=True).groups.items():
                actual, prediction = pred.loc[idx, 'actual'], pred.loc[idx, name]
                metrics = evaluate(actual, prediction) if len(idx) > 1 else {
                    'RMSE': abs(float(prediction.iloc[0]-actual.iloc[0])),
                    'MAE': abs(float(prediction.iloc[0]-actual.iloc[0])), 'R2': np.nan,
                    'Mean Error': float(prediction.iloc[0]-actual.iloc[0])}
                groups.append(dict(fold=fold, configuration=name, dimension=dimension, group=str(label),
                                   rows=len(idx), row_share_pct=100*len(idx)/len(valid),
                                   sse_share_pct=100*sq.loc[idx].sum()/sq.sum(),
                                   actual_mean=actual.mean(), predicted_mean=prediction.mean(),
                                   small_group=len(idx)<100, **metrics))
        monthly_bias = error.groupby(p.month).transform('mean')
        # Oracle quantities describe additive-offset headroom, not deployable scores.
        mse = sq.mean()
        oracles.append(dict(fold=fold, configuration=name, original_rmse=np.sqrt(mse),
                            oracle_global_rmse=np.sqrt(np.mean((error-error.mean())**2)),
                            oracle_monthly_rmse=np.sqrt(np.mean((error-monthly_bias)**2)),
                            monthly_bias_mse_share_pct=100*np.mean(monthly_bias**2)/mse,
                            largest_1pct_error_sse_share_pct=100*sq.nlargest(max(1, int(np.ceil(.01*len(sq))))).sum()/sq.sum()))
    return overall, groups, oracles, edges


def saved_formal(valid):
    base = ROOT/'experiments/feature_engineering/predictions'
    e2 = pd.read_csv(base/'e2_validation_predictions.csv', index_col='listing_index')
    e3 = pd.read_csv(base/'e3_validation_predictions.csv', index_col='listing_index')
    inc = pd.read_csv(base/'incremental_predictions.csv', index_col='listing_index')
    for frame in [e2, e3, inc]:
        validate_predictions(frame, valid)
    return pd.DataFrame({'actual': e3.actual, 'F0': e2['E2.0'], 'Core': e3['E3.0'],
                         'HDB': e3['E3.1'], 'HDB_MRT': e3['E3.2'], 'Schools': e3['E3.3'],
                         'Malls': inc.D1, 'COE': inc.D2, 'Stock': inc.D3, 'All': e3['E3.4']})


def plots(overall, group):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    target = DEST/'figures'
    target.mkdir(exist_ok=True)
    colors = dict(zip(['Core', 'HDB', 'HDB_MRT', 'Schools', 'Malls'], ['#777777', '#CC79A7', '#0072B2', '#009E73', '#D55E00']))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, fold in zip(axes, overall.fold.unique()):
        rows = group[(group.fold == fold) & (group.dimension == 'horizon_band')]
        for name, color in colors.items():
            r = rows[rows.configuration == name].sort_values('group')
            labels = r.group.replace({'09-12': '09-11'}) if fold == 'formal_2024_04' else r.group
            ax.plot(labels, r['Mean Error'], marker='o', label=name, color=color)
        ax.axhline(0, color='black', lw=.7)
        title = 'Formal 2024-04' if fold == 'formal_2024_04' else fold.replace('origin_', 'Origin ').replace('_', '-')
        ax.set(title=title, xlabel='Months ahead', ylabel='Mean error (SGD)')
    axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(target/'horizon_bias.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, color in colors.items():
        r = overall[overall.configuration == name]
        ax.plot(r.fold, r.RMSE, marker='o', label=name, color=color)
    ax.set(ylabel='RMSE (SGD)', title='Feature configurations across forecast origins')
    ax.legend(ncol=2); fig.tight_layout(); fig.savefig(target/'fold_rmse.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(CONTROLS))
    for j, fold in enumerate(overall.fold.unique()):
        score = overall[overall.fold == fold].set_index('configuration').RMSE
        delta = [score[control] - score[candidate] for candidate, control in CONTROLS.items()]
        ax.barh(positions + (j-1)*.24, delta, height=.24, label=fold)
    ax.set_yticks(positions, [f'{candidate} vs {control}' for candidate, control in CONTROLS.items()])
    ax.axvline(0, color='black', lw=.8)
    ax.set(xlabel='RMSE improvement (SGD; positive is better)', title='Incremental feature gains depend on forecast origin')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(target/'feature_deltas.png', dpi=180); plt.close(fig)
    r = group[(group.fold == 'formal_2024_04') & (group.configuration == 'Schools') & (group.dimension == 'TOWN')].sort_values('sse_share_pct', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 4))
    pos = np.arange(len(r))
    ax.bar(pos-.18, r.row_share_pct, .36, label='Share of rows', color='#0072B2')
    ax.bar(pos+.18, r.sse_share_pct, .36, label='Share of squared error', color='#D55E00')
    ax.set_xticks(pos, r.group, rotation=35, ha='right')
    ax.set(ylabel='Percent', title='E3.3: error contribution of ten largest-contributing towns')
    ax.legend(); fig.tight_layout(); fig.savefig(target/'town_error_contribution.png', dpi=180); plt.close(fig)


def main():
    for name in ['results', 'predictions']:
        (DEST/name).mkdir(exist_ok=True)
    source_files = [ROOT/'data/train.csv', *sorted((ROOT/'data/auxiliary').glob('*.csv'))]
    official_files = [*sorted((ROOT/'experiments/feature_engineering/results').rglob('*.csv')),
                      *sorted((ROOT/'experiments/feature_engineering/predictions').glob('*.csv'))]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protected = {str(p.relative_to(ROOT)): digest(p) for p in official_files}
    train, valid = load_data()
    auxiliary, _ = prepare_auxiliary()
    # Deterministic static joins and calendar macro aggregations; no rental labels used.
    core = make_listing(train)
    features = {name: core[F0] if groups is None else build(core[CORE], auxiliary, groups)
                for name, groups in CONFIGS.items()}
    totals, group_rows, oracle_rows, fold_audits = [], [], [], []
    for fold, cutoff, horizon in FOLDS:
        a, b = split_at(train, cutoff, horizon)
        prediction = pd.DataFrame({'actual': b.MONTHLY_RENT})
        for name, frame in features.items():
            x, z, _ = encode(frame.loc[a.index], frame.loc[b.index], 'ordinal')
            assert x.index.equals(a.index) and z.index.equals(b.index)
            assert x.columns.equals(z.columns)
            model = LGBMRegressor(**PARAMS)
            model.fit(x, a.MONTHLY_RENT)
            prediction[name] = model.predict(z)
            print(f'{fold} {name}: RMSE {evaluate(b.MONTHLY_RENT, prediction[name])["RMSE"]:.3f}', flush=True)
        prediction.to_csv(DEST/'predictions'/f'{fold}.csv.gz', index_label='listing_index',
                          compression={'method': 'gzip', 'mtime': 0})
        result, grouped, oracle, edges = summarize(fold, a, b, cutoff, prediction)
        totals.extend(result); group_rows.extend(grouped); oracle_rows.extend(oracle)
        fold_audits.append(dict(fold=fold, cutoff=cutoff, horizon=horizon, train_rows=len(a),
                               validation_rows=len(b), validation_end=b.RENT_APPROVAL_DATE.max(),
                               actual_rent_group_edges=[str(x) for x in edges]))
    formal = saved_formal(valid)
    result, grouped, oracle, edges = summarize('formal_2024_04', train, valid, '2024-04', formal)
    totals.extend(result); group_rows.extend(grouped); oracle_rows.extend(oracle)
    fold_audits.append(dict(fold='formal_2024_04', cutoff='2024-04', horizon=11,
                           train_rows=len(train), validation_rows=len(valid),
                           actual_rent_group_edges=[str(x) for x in edges], predictions='reused, not refit'))
    overall, group = pd.DataFrame(totals), pd.DataFrame(group_rows)
    pairs = []
    for fold in overall.fold.unique():
        scores = overall[overall.fold == fold].set_index('configuration')
        monthly = group[(group.fold == fold) & (group.dimension == 'month')]
        for candidate, control in CONTROLS.items():
            left = monthly[monthly.configuration == candidate].set_index('group')
            right = monthly[monthly.configuration == control].set_index('group')
            pairs.append(dict(fold=fold, candidate=candidate, control=control,
                              rmse_improvement=scores.loc[control, 'RMSE']-scores.loc[candidate, 'RMSE'],
                              mae_improvement=scores.loc[control, 'MAE']-scores.loc[candidate, 'MAE'],
                              months_rmse_improved=int((left.RMSE < right.RMSE).sum()), months=len(left)))
    overall.to_csv(DEST/'results/overall.csv', index=False)
    group.to_csv(DEST/'results/group_errors.csv', index=False)
    pd.DataFrame(pairs).to_csv(DEST/'results/paired_comparisons.csv', index=False)
    pd.DataFrame(oracle_rows).to_csv(DEST/'results/oracle_diagnostics.csv', index=False)
    plots(overall, group)
    assert protected == {str(p.relative_to(ROOT)): digest(p) for p in official_files}
    audit = dict(model_params=PARAMS, encodings={'low_card': 'onehot', 'high_card': 'ordinal_numeric'},
                 feature_sets={name: list(frame.columns) for name, frame in features.items()},
                 folds=fold_audits, data_hashes={str(p.relative_to(ROOT)): digest(p) for p in source_files},
                 official_outputs_unchanged=True, test_csv_used=False, E4_used=False,
                 python=platform.python_version(),
                 versions={p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'lightgbm', 'scikit-learn', 'matplotlib']},
                 limitations=[
                     'Historical validation windows overlap; no independent-fold significance claims.',
                     'Formal validation was already used in E2/E3 selection. Historical checks are retrospective robustness diagnostics, not fresh independent test data.',
                     'Encodings are fixed from previous E2 selection, then mappings are fitted separately on each fold training set.',
                     'Macro diagnostics assume supplied evaluation-period covariates are permitted; not a strict as-of-origin information set. One calendar-month lag is retained.',
                     'Static geographic snapshots are not historically reconstructed.',
                     'Group bins use fixed area edges or training-only rent quantiles/address counts. True-rent and largest-error groups are diagnostic only.',
                     'R2 undefined for singleton groups; small groups flagged below 100 observations.',
                     'Two historical origins have shorter training histories and different market regimes than final training; differences cannot be attributed solely to horizon.',
                 ])
    (DEST/'results/audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('E5 finished; official outputs unchanged.', flush=True)


if __name__ == '__main__':
    main()
