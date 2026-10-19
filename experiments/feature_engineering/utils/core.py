"""Shared encoding, fixed model and E2 runners for feature experiments."""
from pathlib import Path
import json
import platform
import time
import hashlib
import importlib.metadata

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features import (make_listing, feature_groups, ordinal_encoding,
                          onehot_encoding, encode_high_card, make_auxiliary)
from src.auxiliary import load_auxiliary, attach_auxiliary
from src.validation import load_data
from src.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments' / 'feature_engineering'
OUT = BASE / 'results'
AUDITS = BASE / 'audits'
PREDICTIONS = BASE / 'predictions'
FIGURES = BASE / 'figures'
for folder in [OUT, AUDITS, PREDICTIONS, FIGURES]:
    folder.mkdir(parents=True, exist_ok=True)
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
              random_state=42, n_jobs=-1, verbosity=-1)
F0 = ['month_index', 'flat_age', 'FLOOR_AREA_SQM', 'FLAT_TYPE', 'TOWN']
# Preserve make_listing's canonical order for the full core and reference.
CORE = ['FLOOR_AREA_SQM', 'flat_age', 'month_index', 'TOWN', 'FLAT_TYPE',
        'FLAT_MODEL', 'BLOCK', 'STREET']
E2_FEATURES = {
    'E2.0': F0,
    'E2.1': F0 + ['FLAT_MODEL'],
    'E2.2': F0 + ['STREET'],
    'E2.3': F0 + ['composite_block_id'],
    'E2.4': F0 + ['LATITUDE', 'LONGITUDE'],
    'E2.5': CORE,
}


def data():
    train, valid = load_data()
    assert (len(train), len(valid)) == (117777, 31582)
    assert train.index.intersection(valid.index).empty
    return make_listing(train), make_listing(valid), train.MONTHLY_RENT, valid.MONTHLY_RENT


def encode(a, b, method, reference=False):
    _, low, high = feature_groups(a)
    high += [c for c in ['composite_block_id'] if c in a]
    if reference:
        a, b = ordinal_encoding(a, b, columns=low)
    elif low:
        a, b = onehot_encoding(a, b, columns=low)
    if high:
        a, b = encode_high_card(a, b, method=method, columns=high)
    return a, b, high


def fit(name, a, b, y, v, method, reference=False):
    assert a.index.equals(y.index) and b.index.equals(v.index)
    assert a.columns.equals(b.columns)
    assert 'MONTHLY_RENT' not in a
    start = time.perf_counter()
    x, z, high = encode(a, b, method, reference)
    encoding_seconds = time.perf_counter() - start
    assert x.index.equals(y.index) and z.index.equals(v.index)
    assert x.columns.equals(z.columns)
    for frame in [x, z]:
        numeric = frame.select_dtypes(include='number')
        assert not np.isinf(numeric.to_numpy(dtype=float)).any()
        assert all(pd.api.types.is_numeric_dtype(t) or isinstance(t, pd.CategoricalDtype)
                   for t in frame.dtypes)
    native = [c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)]
    assert native == (high if method == 'native' else [])
    model = LGBMRegressor(**PARAMS)
    start = time.perf_counter()
    model.fit(x, y, categorical_feature=native if native else 'auto')
    seconds = time.perf_counter() - start
    assert len(model.booster_.pandas_categorical or []) == len(native)
    pred = model.predict(z)
    metrics = evaluate(v, pred)
    assert np.isfinite(pred).all() and np.isfinite(list(metrics.values())).all()
    row = dict(experiment=name, encoding=method, low_card_encoding='ordinal' if reference else 'onehot',
               raw_features=json.dumps(list(a)), raw_feature_count=a.shape[1],
               encoded_features=json.dumps(list(x)), feature_count=x.shape[1],
               categorical_features=json.dumps(native), model_params=json.dumps(PARAMS, sort_keys=True),
               train_rows=len(x), validation_rows=len(z), encoding_seconds=encoding_seconds,
               training_seconds=seconds, **metrics)
    print(name, method, {k: round(v, 6) for k, v in metrics.items()}, flush=True)
    return row, pred


def save(rows, name, controls):
    table = pd.DataFrame(rows)
    rmse = table.set_index('experiment').RMSE
    table['control'] = [controls[r.experiment] for r in table.itertuples()]
    table['delta_rmse'] = [rmse[c] - r for c, r in zip(table.control, table.RMSE)]
    table['relative_improvement_pct'] = [100*d/rmse[c] for d, c in zip(table.delta_rmse, table.control)]
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / name, index=False)
    return table


def stage1():
    OUT.mkdir(exist_ok=True)
    a, b, y, v = data()
    row, pred = fit('Listing reference', a, b, y, v, 'ordinal', reference=True)
    pd.DataFrame([row]).to_csv(OUT/'listing_reference.csv', index=False)
    stored = pd.read_csv(ROOT/'experiments/model_comparison_results.csv')
    prior = float(stored.loc[stored.Model.eq('LightGBM'), 'RMSE'].iloc[0])
    audit = dict(notebook_rmse=480.60, csv_rmse=prior, reproduced_rmse=row['RMSE'],
                 notebook_match_2dp=round(row['RMSE'], 2) == 480.60,
                 csv_match=bool(np.isclose(row['RMSE'], prior)),
                 python=platform.python_version(),
                 versions={p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm']},
                 train_file_sha256=hashlib.sha256((ROOT/'data/train.csv').read_bytes()).hexdigest())
    (AUDITS/'reference_audit.json').write_text(json.dumps(audit, indent=2))
    rows=[]
    for method in ['ordinal','frequency','native']:
        r, _ = fit(method, a[CORE], b[CORE], y, v, method)
        rows.append(r)
    table = save(rows, 'encoding_comparison.csv', {m:'ordinal' for m in ['ordinal','frequency','native']})
    winner = table.loc[table.RMSE.idxmin(), 'encoding']
    (AUDITS/'selected_encoding.json').write_text(json.dumps({'method':winner, 'selection':'Lowest validation RMSE on E2.5; fixed for all E2 and future E3.'}, indent=2))
    plot(table, stem='encoding_rmse', title='E2.5: High-cardinality encoding comparison')
    return table, audit


def composite(frame):
    # JSON tuples avoid separator collisions and distinguish nulls from literal strings.
    def key(values):
        return json.dumps([None if pd.isna(v) or str(v).lower() == 'nan' else str(v)
                           for v in values], ensure_ascii=True, separators=(',', ':'))
    return frame[['TOWN','BLOCK','STREET']].apply(lambda r: key(r.tolist()), axis=1)


def stage2():
    method = json.loads((AUDITS/'selected_encoding.json').read_text())['method']
    a, b, y, v = data()
    a['composite_block_id'], b['composite_block_id'] = composite(a), composite(b)
    raw = load_auxiliary()
    auxiliary = make_auxiliary(df_hdb=raw['hdb'])
    enriched=[]
    for frame in [a, b]:
        joined=attach_auxiliary(frame, auxiliary)
        assert joined.index.equals(frame.index) and len(joined)==len(frame)
        # E2.4 has coordinates only, never other HDB properties.
        enriched.append(joined)
    rows=[]
    predictions={}
    for name, columns in E2_FEATURES.items():
        aa, bb = enriched if name=='E2.4' else (a,b)
        r, pred = fit(name, aa[columns], bb[columns], y, v, method)
        rows.append(r)
        predictions[name]=pred
    table=save(rows, 'e2_results.csv', {n:'E2.0' for n in E2_FEATURES})
    comparison=pd.read_csv(OUT/'encoding_comparison.csv')
    expected=comparison.loc[comparison.encoding.eq(method)].iloc[0]
    current=table.loc[table.experiment.eq('E2.5')].iloc[0]
    for metric in ['RMSE','MAE','R2','Mean Error']:
        assert np.isclose(current[metric], expected[metric], rtol=0, atol=1e-9)
    pd.DataFrame({'actual':v, **predictions}, index=v.index).to_csv(PREDICTIONS/'e2_validation_predictions.csv', index_label='listing_index')
    plot(table)
    return table


def plot(table, stem='e2_rmse', title='E2: Listing feature experiments'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(7.2,3.6))
    ax.bar(table.experiment, table.RMSE, color='#487a9b', width=.6)
    for i,r in enumerate(table.RMSE): ax.annotate(f'{r:.2f}', (i,r), xytext=(0,4), textcoords='offset points', ha='center', fontsize=9)
    ax.set(ylabel='Validation RMSE (SGD)', title=title, ylim=(0,table.RMSE.max()*1.12))
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(FIGURES/f'{stem}.{ext}', dpi=200)
    plt.close(fig)
