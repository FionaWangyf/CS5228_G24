"""E3 only: reuse E2 core/model/encoders and shared auxiliary engineering."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd

from experiments.feature_engineering.utils.core import ROOT, OUT, AUDITS, PREDICTIONS, FIGURES, CORE, PARAMS, data, encode, fit
from src.auxiliary import load_auxiliary, attach_auxiliary
from src.features import make_auxiliary

EXPERIMENTS = {
    'E3.0': (),
    'E3.1': ('hdb',),
    'E3.2': ('hdb', 'mrt'),
    'E3.3': ('hdb', 'mrt', 'schools'),
    'E3.4': ('hdb', 'mrt', 'schools', 'malls', 'coe', 'stock'),
}
GROUP_FEATURES = {
    'hdb': ['LATITUDE', 'LONGITUDE', 'MAX_FLOOR', 'SUBZONE', 'PLANNING_AREA'],
    'mrt': ['mrt_distance_km', 'mrt_name', 'mrt_line_count'],
    'schools': ['primary_school_distance_km', 'secondary_school_distance_km',
                'schools_within_1km', 'schools_within_2km'],
    'malls': ['mall_distance_km', 'mall_name'],
    'coe': ['coe_sum'],
    'stock': ['stock_index'],
}
METRICS = ['RMSE', 'MAE', 'R2', 'Mean Error']
RESULTS = OUT


def lag_monthly_auxiliary(auxiliary, months=1):
    """Copy tables; map source calendar month s to listing month s + months.

    This shifts keys, not rows, so a missing month remains missing. No forward
    or backward filling is applied. Shared make/attach defaults stay unchanged.
    """
    if isinstance(months, bool) or not isinstance(months, int) or months < 1:
        raise ValueError('Use a positive integer calendar-month lag.')
    out = {name: table.copy() for name, table in auxiliary.items()}
    for name in ('coe', 'stock'):
        if name in out:
            keys = out[name]['month_index']
            if keys.isna().any() or keys.duplicated().any() or not (keys % 1 == 0).all():
                raise ValueError(f'{name} needs unique integer source month keys.')
            out[name]['month_index'] = keys + months
    return out


def prepare_auxiliary():
    """Inspect local schema and aggregate only dates needed by lagged joins."""
    raw = load_auxiliary()
    audit = {
        'monthly_lag': 1,
        'availability_assumption': 'The previous complete calendar month is available at prediction time; exact release dates are not recorded.',
        'geographic_policy': 'Static Geographic Approximation; no historical POI reconstruction.',
        'source_schema': {k: list(v.columns) for k, v in raw.items() if k != 'mrt_open'},
        'raw_coe_range': [raw['coe'].YEAR_MONTH.min(), raw['coe'].YEAR_MONTH.max()],
        'raw_stock_range': [raw['stock'].DATE.min(), raw['stock'].DATE.max()],
    }
    # Last validation month is 2025-03, hence no source later than 2025-02 is used.
    coe = raw['coe'].loc[raw['coe'].YEAR_MONTH <= '2025-02'].copy()
    stock = raw['stock'].loc[raw['stock'].DATE.str[:7] <= '2025-02'].copy()
    rounds = coe.groupby(['YEAR_MONTH', 'CATEGORY']).ROUND.nunique().unstack()
    assert set(rounds.columns) == {'a', 'b', 'c', 'e'}
    assert (rounds == 2).all().all(), 'Incomplete COE rounds in required source period.'
    monthly = stock.pivot_table(index=stock.DATE.str[:7], columns='SYMBOL',
                               values='ADJUSTED_CLOSE', aggfunc='mean').sort_index()
    universe = monthly.columns[monthly.iloc[0].notna() & monthly.columns.str.endswith('.SI')]
    audit['stock_universe'] = list(universe)
    audit['stock_months_missing_entire_symbol'] = int(monthly[universe].isna().sum().sum())
    audit['stock_missing_adjusted_close_raw_rows'] = int(pd.read_csv(
        ROOT/'data/auxiliary/sg-stock-prices.csv', usecols=['ADJUSTED_CLOSE'])
        .ADJUSTED_CLOSE.isna().sum())
    audit['stock_observation_range_used'] = [stock.DATE.min(), stock.DATE.max()]
    audit['coe_source_month_range_used'] = [coe.YEAR_MONTH.min(), coe.YEAR_MONTH.max()]
    auxiliary = make_auxiliary(df_coe=coe, df_stock=stock, df_hdb=raw['hdb'],
                               df_mrt=raw['mrt'], df_schools=raw['schools'], df_malls=raw['malls'])
    lagged = lag_monthly_auxiliary(auxiliary)
    expected = set(range(0, 50))  # 2021-01 through 2025-02
    mapping = []
    for name in ['coe', 'stock']:
        source = auxiliary[name]
        assert set(source.month_index) == expected
        assert source.drop(columns='month_index').notna().all().all()
        pd.testing.assert_frame_equal(source.drop(columns='month_index'), lagged[name].drop(columns='month_index'))
        assert (lagged[name].month_index.to_numpy() - source.month_index.to_numpy() == 1).all()
        for s, t in zip(source.month_index, lagged[name].month_index):
            mapping.append({'table': name, 'source_month_index': s, 'listing_month_index': t,
                            'source_month': str(pd.Period('2021-01', freq='M') + int(s)),
                            'listing_month': str(pd.Period('2021-01', freq='M') + int(t))})
    audit['monthly_source_mapping'] = mapping
    return lagged, audit


def build(core, auxiliary, groups):
    """Whitelist precisely the selected groups; preserve canonical core order."""
    joined = attach_auxiliary(core, {name: auxiliary[name] for name in groups})
    expected = CORE + [c for group in groups for c in GROUP_FEATURES[group]]
    assert set(joined.columns) == set(expected), 'Unexpected auxiliary columns.'
    assert joined.index.equals(core.index) and len(joined) == len(core)
    pd.testing.assert_frame_equal(joined[CORE], core[CORE])
    return joined[expected]


def run_e3():
    """Train exactly five E3 models; abort after E3.0 if E2.5 differs."""
    e2 = pd.read_csv(OUT/'e2_results.csv').set_index('experiment').loc['E2.5']
    method = json.loads((AUDITS/'selected_encoding.json').read_text())['method']
    assert method == e2.encoding == 'ordinal'
    assert e2.low_card_encoding == 'onehot'
    assert json.loads(e2.model_params) == PARAMS
    assert json.loads(e2.raw_features) == CORE
    a, b, y, v = data()
    a, b = a[CORE], b[CORE]
    prior_audit = json.loads((AUDITS/'reference_audit.json').read_text())
    assert hashlib.sha256((ROOT/'data/train.csv').read_bytes()).hexdigest() == prior_audit['train_file_sha256']
    auxiliary, audit = prepare_auxiliary()
    rows, predictions, nulls = [], {}, []
    baseline_audit = {}
    for name, groups in EXPERIMENTS.items():
        x, z = build(a, auxiliary, groups), build(b, auxiliary, groups)
        row, pred = fit(name, x, z, y, v, method)
        row['feature_set'] = 'Core Only' if not groups else 'Core + ' + ' + '.join(g.upper() if g in ['hdb','mrt','coe'] else g.title() for g in groups)
        row['auxiliary_groups'] = json.dumps(groups)
        row['monthly_lag'] = 1 if 'coe' in groups else None
        if name == 'E3.0':
            assert row['raw_feature_count'] == e2.raw_feature_count
            assert row['feature_count'] == e2.feature_count
            assert row['encoded_features'] == e2.encoded_features
            assert row['train_rows'] == e2.train_rows and row['validation_rows'] == e2.validation_rows
            for metric in METRICS:
                diff = row[metric] - float(e2[metric])
                assert np.isclose(diff, 0, rtol=0, atol=1e-9), f'E3.0 mismatch: {metric}'
                baseline_audit[metric + '_difference'] = diff
            saved = pd.read_csv(PREDICTIONS/'e2_validation_predictions.csv', index_col='listing_index')
            assert saved.index.equals(v.index)
            np.testing.assert_allclose(saved.actual, v, rtol=0, atol=0)
            np.testing.assert_allclose(saved['E2.5'], pred, rtol=0, atol=1e-9)
            baseline_audit['max_prediction_difference'] = float(np.max(np.abs(saved['E2.5'].to_numpy()-pred)))
            baseline_audit['passed'] = True
        for split, frame in [('train', x), ('validation', z)]:
            for column in frame:
                nulls.append(dict(experiment=name, split=split, feature=column,
                                  missing_rows=int(frame[column].isna().sum()), rows=len(frame)))
            if name == 'E3.4':
                for column in ['coe_sum', 'stock_index']:
                    # No 2020-12 source: only initial training month is missing.
                    np.testing.assert_array_equal(frame[column].isna(), frame.month_index.eq(0))
        rows.append(row)
        predictions[name] = pred
    table = pd.DataFrame(rows)
    table['control'] = ['E3.0', 'E3.0', 'E3.1', 'E3.2', 'E3.3']
    control_rmse = table.set_index('experiment').RMSE
    table['delta_rmse'] = [float(control_rmse[c]) - r for c, r in zip(table.control, table.RMSE)]
    table['relative_improvement_pct'] = [100*d/float(control_rmse[c]) for c, d in zip(table.control, table.delta_rmse)]
    table['delta_rmse_vs_core'] = table.iloc[0].RMSE - table.RMSE
    table['relative_improvement_vs_core_pct'] = 100*table.delta_rmse_vs_core/table.iloc[0].RMSE
    table.to_csv(RESULTS/'e3_results.csv', index=False)
    pd.DataFrame({'actual': v, **predictions}, index=v.index).to_csv(PREDICTIONS/'e3_validation_predictions.csv', index_label='listing_index')
    pd.DataFrame(nulls).to_csv(AUDITS/'e3_missingness.csv', index=False)
    audit['e30_vs_e25'] = baseline_audit
    audit['python'] = platform.python_version()
    audit['versions'] = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm']}
    audit['train_file_sha256'] = prior_audit['train_file_sha256']
    audit['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data/auxiliary').glob('*.csv'))}
    (AUDITS/'e3_data_audit.json').write_text(json.dumps(audit, indent=2))
    return table, audit



def plot_e3(table):
    """Refresh the E3 figure from results; no model fitting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(table.experiment, table.RMSE, color='#487a9b', width=.6)
    for i, value in enumerate(table.RMSE):
        ax.annotate(f'{value:.2f}', (i, value), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
    ax.set(ylabel='Validation RMSE (SGD)', title='E3: Auxiliary feature integration', ylim=(0, table.RMSE.max()*1.12))
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(FIGURES/f'e3_rmse_comparison.{ext}', dpi=200)
    plt.close(fig)
