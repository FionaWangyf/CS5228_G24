"""Run the predeclared E6 A–D integration and matched residual analysis.

Usage: .venv/bin/python -m experiments.e6.run_e6
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.auxiliary import load_auxiliary
from src.features import make_listing, make_auxiliary, feature_groups, onehot_encoding, encode_high_card
from src.validation import load_data
from src.evaluation import evaluate
from experiments.feature_engineering.utils.core import ROOT, CORE, PARAMS
from experiments.feature_engineering.utils.auxiliary import build
from experiments.error_analysis.e5 import FOLDS, split_at, profile, validate_predictions

DEST = Path(__file__).resolve().parent
CONFIGS = {
    'A': ('hdb', 'mrt', 'schools', 'malls'),
    'B': ('hdb', 'mrt', 'malls'),
    'C': ('hdb', 'mrt', 'malls'),
    'D': ('hdb', 'malls'),
}
PAIRS = [('B', 'A', 'remove_schools'), ('C', 'B', 'replace_time'),
         ('D', 'B', 'remove_mrt'), ('C', 'A', 'combined_vs_reference'),
         ('D', 'A', 'combined_vs_reference')]
DIMENSIONS = ['month', 'horizon_band', 'TOWN', 'FLAT_TYPE', 'area_band', 'address_support', 'actual_rent_band']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_config(frame, auxiliary, name):
    out = build(make_listing(frame)[CORE], auxiliary, CONFIGS[name])
    if name == 'C':
        dates = pd.to_datetime(frame.RENT_APPROVAL_DATE, format='%Y-%m')
        out = out.rename(columns={'month_index': 'year'})
        out['year'] = dates.dt.year
        out['calendar_month'] = dates.dt.month.astype(str)
        assert 'month_index' not in out
    return out


def encode_config(train, valid):
    _, low, high = feature_groups(train)
    if 'calendar_month' in train:
        low = low + ['calendar_month']
    x, z = onehot_encoding(train, valid, columns=low)
    x, z = encode_high_card(x, z, method='ordinal', columns=high)
    assert x.index.equals(train.index) and z.index.equals(valid.index)
    assert x.columns.equals(z.columns)
    assert 'MONTHLY_RENT' not in x
    assert all(pd.api.types.is_numeric_dtype(t) for t in x.dtypes)
    return x, z


def score(actual, predicted):
    metrics = evaluate(actual, predicted) if len(actual) > 1 else {
        'RMSE': abs(float(predicted.iloc[0]-actual.iloc[0])),
        'MAE': abs(float(predicted.iloc[0]-actual.iloc[0])),
        'R2': np.nan, 'Mean Error': float(predicted.iloc[0]-actual.iloc[0])}
    return metrics


def analyze(fold, train, valid, cutoff, pred):
    validate_predictions(pred, valid)
    prof, edges = profile(train, valid, cutoff)
    overall, grouped, paired, oracle = [], [], [], []
    for name in CONFIGS:
        err = pred[name] - pred.actual
        sq = err**2
        overall.append(dict(fold=fold, configuration=name, train_rows=len(train), validation_rows=len(valid),
                            cutoff=cutoff, validation_end=valid.RENT_APPROVAL_DATE.max(), **score(pred.actual, pred[name])))
        for dim in DIMENSIONS:
            for label, idx in prof.groupby(dim, observed=True).groups.items():
                grouped.append(dict(fold=fold, configuration=name, dimension=dim, group=str(label), rows=len(idx),
                                    row_share_pct=100*len(idx)/len(valid), sse_share_pct=100*sq.loc[idx].sum()/sq.sum(),
                                    actual_mean=pred.loc[idx,'actual'].mean(), predicted_mean=pred.loc[idx,name].mean(),
                                    small_group=len(idx)<100, **score(pred.loc[idx,'actual'], pred.loc[idx,name])))
        bias = err.groupby(prof.month).transform('mean')
        oracle.append(dict(fold=fold, configuration=name, RMSE=np.sqrt(sq.mean()),
                           oracle_monthly_rmse=np.sqrt(np.mean((err-bias)**2)),
                           monthly_bias_mse_share_pct=100*np.mean(bias**2)/sq.mean(),
                           largest_1pct_error_sse_share_pct=100*sq.nlargest(int(np.ceil(.01*len(sq)))).sum()/sq.sum()))
    for candidate, control, purpose in PAIRS:
        ec, eb = pred[candidate]-pred.actual, pred[control]-pred.actual
        partitions = [('all', 'all', pred.index)] + [
            (dim, str(label), idx) for dim in DIMENSIONS
            for label, idx in prof.groupby(dim, observed=True).groups.items()]
        for dim, label, idx in partitions:
            c, b = ec.loc[idx], eb.loc[idx]
            paired.append(dict(fold=fold, candidate=candidate, control=control, purpose=purpose,
                               dimension=dim, group=label, rows=len(idx), small_group=len(idx)<100,
                               rmse_improvement=np.sqrt(np.mean(b**2))-np.sqrt(np.mean(c**2)),
                               mae_improvement=np.mean(abs(b))-np.mean(abs(c)),
                               mean_error_change=c.mean()-b.mean(),
                               abs_bias_improvement=abs(b.mean())-abs(c.mean()),
                               fraction_rows_lower_absolute_error=float((abs(c)<abs(b)).mean()),
                               mse_improvement=np.mean(b**2-c**2),
                               sse_improvement=float(np.sum(b**2-c**2))))
    return overall, grouped, paired, oracle, edges


def reuse_a(fold, valid):
    if fold == 'formal_2024_04':
        path = ROOT/'experiments/feature_engineering/predictions/incremental_predictions.csv'
        column = 'D1'
        metrics_path = ROOT/'experiments/feature_engineering/results/diagnostics/incremental_results.csv'
        expected = pd.read_csv(metrics_path).set_index('experiment').loc['D1']
    else:
        path = ROOT/'experiments/error_analysis/predictions'/f'{fold}.csv.gz'
        column = 'Malls'
        table = pd.read_csv(ROOT/'experiments/error_analysis/results/overall.csv')
        expected = table[(table.fold == fold) & (table.configuration == 'Malls')].iloc[0]
    saved = pd.read_csv(path, index_col='listing_index')
    validate_predictions(saved, valid)
    for metric, value in score(saved.actual, saved[column]).items():
        np.testing.assert_allclose(value, expected[metric], rtol=0, atol=1e-9)
    return saved[column], {'source': str(path.relative_to(ROOT)), 'sha256': digest(path), 'column': column}


def make_plots(overall, groups, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = DEST/'figures'
    out.mkdir(exist_ok=True)
    colors = {'A':'#777777', 'B':'#0072B2', 'C':'#D55E00', 'D':'#009E73'}
    totals = pairs[pairs.dimension=='all']
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    comparisons = [('B','A'),('C','B'),('D','B')]
    titles=['B vs A: remove Schools','C vs B: replace time','D vs B: remove MRT']
    for ax,(c,b),title in zip(axes,comparisons,titles):
        values=[totals[(totals.fold==fold)&(totals.candidate==c)&(totals.control==b)].rmse_improvement.iloc[0]
                for fold in overall.fold.unique()]
        ax.barh(np.arange(3),values,color=['#0072B2','#E69F00','#009E73'])
        ax.set_yticks(np.arange(3),['2022-06','2022-12','2024-04'])
        for i,value in enumerate(values):
            ax.annotate(f'{value:+.3f}',(value,i),xytext=(4 if value>=0 else -4,0),textcoords='offset points',
                        ha='left' if value>=0 else 'right',va='center',fontsize=9)
        ax.margins(x=.35);ax.axvline(0,color='black',lw=.8)
        ax.set(title=title,xlabel='RMSE improvement (SGD)',ylabel='Forecast origin')
    fig.suptitle('Positive is better; each panel has its own scale',fontsize=10)
    fig.tight_layout();fig.savefig(out/'paired_rmse.png',dpi=180);plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(10,7),sharex=True)
    rows=groups[(groups.fold=='formal_2024_04')&(groups.dimension=='month')]
    for name,color in colors.items():
        r=rows[rows.configuration==name].sort_values('group')
        for ax,metric in zip(axes,['RMSE','Mean Error']):
            ax.plot(r.group,r[metric],marker='o',label=name,color=color);ax.set_ylabel(metric+' (SGD)')
    axes[1].axhline(0,color='black',lw=.8);axes[0].legend(ncol=4)
    axes[0].set_title('E6: original validation period (2024-05 to 2025-03)')
    axes[1].tick_params(axis='x',rotation=35);fig.tight_layout();fig.savefig(out/'formal_monthly_errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,fold in zip(axes,overall.fold.unique()):
        r=groups[(groups.fold==fold)&(groups.dimension=='horizon_band')]
        for name,color in colors.items():
            q=r[r.configuration==name].sort_values('group')
            labels=q.group.replace({'09-12':'09-11'}) if fold=='formal_2024_04' else q.group
            ax.plot(labels,q['Mean Error'],marker='o',label=name,color=color)
        ax.axhline(0,color='black',lw=.8);ax.set(title=fold,xlabel='Months ahead',ylabel='Mean error (SGD)')
    axes[-1].legend();fig.tight_layout();fig.savefig(out/'horizon_bias.png',dpi=180);plt.close(fig)


def main():
    for d in ['results','predictions','figures']:
        (DEST/d).mkdir(exist_ok=True)
    protected_files = [p for folder in ['experiments/feature_engineering','experiments/e4','experiments/error_analysis']
                       for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected = {str(p.relative_to(ROOT)):digest(p) for p in protected_files}
    # Reusing results requires their underlying data and protocol to match.
    prior = json.loads((ROOT/'experiments/error_analysis/results/audit.json').read_text())
    assert prior['model_params']==PARAMS
    for rel, sha in prior['data_hashes'].items():
        assert digest(ROOT/rel)==sha, f'Changed E5 source: {rel}'
    e3audit=json.loads((ROOT/'experiments/feature_engineering/audits/e3_data_audit.json').read_text())
    assert digest(ROOT/'data/train.csv')==e3audit['train_file_sha256']
    for name,sha in e3audit['source_hashes'].items():
        assert digest(ROOT/'data/auxiliary'/name)==sha
    train, valid=load_data()
    raw=load_auxiliary()
    auxiliary=make_auxiliary(df_hdb=raw['hdb'],df_mrt=raw['mrt'],df_schools=raw['schools'],df_malls=raw['malls'])
    folds=[(name,*split_at(train,cutoff,horizon),cutoff) for name,cutoff,horizon in FOLDS]
    folds.append(('formal_2024_04',train,valid,'2024-04'))
    totals,groups,pairs,oracles,audits=[],[],[],[],[]
    for fold,a,b,cutoff in folds:
        pred=pd.DataFrame({'actual':b.MONTHLY_RENT})
        pred['A'], provenance=reuse_a(fold,b)
        detail={'fold':fold,'cutoff':cutoff,'A_provenance':provenance,'fits':{}}
        for name in (['A','B','C','D'] if fold=='formal_2024_04' else ['B','C','D']):
            fit_raw,eval_raw=make_config(a,auxiliary,name),make_config(b,auxiliary,name)
            x,z=encode_config(fit_raw,eval_raw)
            started=time.perf_counter();model=LGBMRegressor(**PARAMS).fit(x,a.MONTHLY_RENT)
            predicted=model.predict(z);seconds=time.perf_counter()-started
            detail['fits'][name]={'raw_features':list(fit_raw),'encoded_features':list(x),
                                  'fit_and_predict_seconds':seconds,'train_rows':len(a),'validation_rows':len(b),
                                  'raw_missing_train':fit_raw.isna().sum().to_dict(),
                                  'raw_missing_validation':eval_raw.isna().sum().to_dict()}
            if name=='A':
                np.testing.assert_allclose(predicted,pred.A,rtol=0,atol=1e-9)
                detail['A_refit_max_difference']=float(np.max(np.abs(predicted-pred.A)))
            else:
                pred[name]=predicted
            print(f'{fold} {name}: {score(b.MONTHLY_RENT,predicted)}',flush=True)
        pred.to_csv(DEST/'predictions'/f'{fold}.csv.gz',index_label='listing_index',compression={'method':'gzip','mtime':0})
        o,g,p,r,edges=analyze(fold,a,b,cutoff,pred)
        totals.extend(o);groups.extend(g);pairs.extend(p);oracles.extend(r)
        detail['rent_edges']=[str(x) for x in edges];audits.append(detail)
    overall,group,paired=pd.DataFrame(totals),pd.DataFrame(groups),pd.DataFrame(pairs)
    for filename,frame in [('overall',overall),('group_errors',group),('paired_errors',paired),('oracle_diagnostics',pd.DataFrame(oracles))]:
        frame.to_csv(DEST/'results'/f'{filename}.csv',index=False)
    monthly=paired[paired.dimension=='month']
    consistency=monthly.groupby(['fold','candidate','control'],as_index=False).agg(
        months=('group','size'),rmse_better_months=('rmse_improvement',lambda s:int((s>0).sum())),
        mae_better_months=('mae_improvement',lambda s:int((s>0).sum())))
    consistency.to_csv(DEST/'results/monthly_consistency.csv',index=False)
    make_plots(overall,group,paired)
    assert protected=={str(p.relative_to(ROOT)):digest(p) for p in protected_files}
    sources=[ROOT/'src/features.py',ROOT/'src/cleaning.py',ROOT/'src/validation.py',ROOT/'src/evaluation.py',
             ROOT/'src/auxiliary.py',ROOT/'experiments/error_analysis/e5.py',
             ROOT/'experiments/feature_engineering/utils/core.py',ROOT/'experiments/feature_engineering/utils/auxiliary.py',
             DEST/'run_e6.py',DEST/'PROTOCOL.md']
    audit=dict(protocol='PROTOCOL.md',model_params=PARAMS,configs=CONFIGS,folds=audits,
               data_hashes=prior['data_hashes'],source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources},
               python=platform.python_version(),versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','scikit-learn','matplotlib']},
               prior_artifacts_unchanged=True,test_csv_used=False,train_fitted_encodings=True,
               original_E4_preprocessing_reproduced=False,
               note='C transfers E4.1 representation to E2/E3 preprocessing. Static geography, retrospective overlapping folds, prior validation selection remain limitations.')
    (DEST/'results/audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('E6 complete. All previous experiment artifacts unchanged.',flush=True)


if __name__=='__main__':
    main()
