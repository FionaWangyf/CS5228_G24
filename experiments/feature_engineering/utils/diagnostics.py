"""Isolated diagnostics. No writes to official experiments, src, or raw data."""
import hashlib
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from experiments.feature_engineering.utils.core import ROOT, OUT, AUDITS, PREDICTIONS, FIGURES, CORE, PARAMS, data, encode, fit, composite
from experiments.feature_engineering.utils.auxiliary import prepare_auxiliary, build, EXPERIMENTS, GROUP_FEATURES
from src.validation import load_data
from src.evaluation import evaluate

DEST = OUT/'diagnostics'
DEST.mkdir(parents=True, exist_ok=True)


def artifact(name):
    if name.endswith('.png'): return FIGURES/name
    if name.endswith('.json'): return AUDITS/name
    if 'predictions' in name: return PREDICTIONS/name
    return DEST/name


def write_json(name, value):
    (artifact(name)).write_text(json.dumps(value, indent=2))


def frames():
    a,b,y,v=data()
    aux,audit=prepare_auxiliary()
    prior=json.loads((AUDITS/'e3_data_audit.json').read_text())
    assert hashlib.sha256((ROOT/'data/train.csv').read_bytes()).hexdigest()==prior['train_file_sha256']
    for file,digest in prior['source_hashes'].items():
        assert hashlib.sha256((ROOT/'data/auxiliary'/file).read_bytes()).hexdigest()==digest
    return a,b,y,v,aux,audit


def prior_predictions(v):
    saved=pd.read_csv(PREDICTIONS/'e3_validation_predictions.csv',index_col='listing_index')
    assert saved.index.equals(v.index)
    np.testing.assert_array_equal(saved.actual, v)
    results=pd.read_csv(OUT/'e3_results.csv').set_index('experiment')
    for name in ['E3.3','E3.4']:
        for m,value in evaluate(v,saved[name]).items():
            assert np.isclose(value, results.loc[name,m], rtol=0,atol=1e-9)
    return saved


def fit_inspected(a,b,y,v,method):
    x,z,high=encode(a,b,method)
    cats=[c for c in x if isinstance(x[c].dtype,pd.CategoricalDtype)]
    assert cats==(high if method=='native' else [])
    model=LGBMRegressor(**PARAMS)
    started=time.perf_counter()
    model.fit(x,y,categorical_feature=cats if cats else 'auto')
    seconds=time.perf_counter()-started
    pred=model.predict(z)
    evaluate(v,pred)
    dump=model.booster_.dump_model()
    splits=[]
    def walk(node):
        if 'split_feature' not in node: return
        splits.append({'feature':model.booster_.feature_name()[node['split_feature']],
                       'decision_type':node['decision_type'], 'gain':node['split_gain']})
        walk(node['left_child']);walk(node['right_child'])
    for tree in dump['tree_info']: walk(tree['tree_structure'])
    return model,x,z,pred,pd.DataFrame(splits),seconds


def encoding_diagnostics():
    a,b,y,v,aux,audit=frames()
    comparison=pd.read_csv(OUT/'encoding_comparison.csv')
    assert set(comparison.encoding)=={'ordinal','frequency','native'}
    for col in ['raw_features','encoded_features','model_params','train_rows','validation_rows','raw_feature_count','feature_count','low_card_encoding']:
        assert comparison[col].nunique()==1, col
    assert json.loads(comparison.iloc[0].raw_features)==CORE
    assert json.loads(comparison.iloc[0].model_params)==PARAMS
    assert comparison.iloc[0].low_card_encoding=='onehot'
    model,x,z,pred,splits,seconds=fit_inspected(a[CORE],b[CORE],y,v,'native')
    expected=comparison.set_index('encoding').loc['native']
    for m,value in evaluate(v,pred).items():
        assert np.isclose(value, expected[m],rtol=0,atol=1e-9)
    cats=['BLOCK','STREET']
    assert model.booster_.params['categorical_column']==[x.columns.get_loc(c) for c in cats]
    assert len(model.booster_.pandas_categorical)==2
    assert set(splits.loc[splits.decision_type=='==','feature'])==set(cats)
    splits.groupby(['feature','decision_type']).agg(split_count=('gain','size'),gain=('gain','sum')).reset_index().to_csv(artifact('native_split_audit.csv'),index=False)
    fulla,fullb=build(a,aux,EXPERIMENTS['E3.4']),build(b,aux,EXPERIMENTS['E3.4'])
    fulla['composite_block_id'],fullb['composite_block_id']=composite(a),composite(b)
    profiles=[]
    for col in ['BLOCK','STREET','SUBZONE','mrt_name','mall_name','composite_block_id']:
        counts=fulla[col].value_counts(dropna=False)
        unknown=~fullb[col].isin(fulla[col].dropna().unique()) & fullb[col].notna()
        profiles.append(dict(feature=col, train_categories=int(fulla[col].nunique()),
                             validation_categories=int(fullb[col].nunique()),
                             train_missing=int(fulla[col].isna().sum()),validation_missing=int(fullb[col].isna().sum()),
                             unknown_validation_rows=int(unknown.sum()),unknown_validation_pct=float(100*unknown.mean()),
                             median_train_rows_per_category=float(counts.median()),
                             train_categories_with_under_20_rows=int((counts<20).sum()),
                             formal_encoding='ordinal_numeric',
                             used_in_encoding_comparison=col in cats))
    pd.DataFrame(profiles).to_csv(artifact('high_cardinality_profile.csv'),index=False)
    ambiguity=a[['TOWN','BLOCK','STREET']].drop_duplicates().groupby('BLOCK').size()
    write_json('encoding_audit.json',dict(
        fair_comparison=True, categorical_dtypes={c:str(x[c].dtype) for c in cats},
        categorical_feature_argument=cats, effective_categorical_column=model.booster_.params['categorical_column'],
        pandas_category_counts=[len(c) for c in model.booster_.pandas_categorical],
        categorical_split_count=int(splits.decision_type.eq('==').sum()),
        categorical_split_features=sorted(splits.loc[splits.decision_type.eq('=='),'feature'].unique()),
        reproduced_metrics=evaluate(v,pred),training_seconds=seconds,
        block_labels_mapping_to_multiple_addresses=int((ambiguity>1).sum()),
        block_labels_total=len(ambiguity),
        scope='Comparison covers BLOCK/STREET on E2.5 only, not auxiliary categories or composite_block_id.',
        recommendation='Keep official encoding unchanged. Native is implemented correctly but was worse on the tested core.'
    ))
    return comparison,pd.DataFrame(profiles)


def temporal_diagnostics():
    a,b,y,v,aux,audit=frames()
    saved=prior_predictions(v)
    train,valid=load_data()
    errors=pd.DataFrame({'month':valid.RENT_APPROVAL_DATE,'actual':v,'predicted':saved['E3.3']})
    rows=[]
    for month,group in errors.groupby('month',sort=True):
        rows.append(dict(month=month,rows=len(group),actual_mean=group.actual.mean(),
                         predicted_mean=group.predicted.mean(),**evaluate(group.actual,group.predicted)))
    monthly=pd.DataFrame(rows)
    assert list(monthly.month)==list(pd.period_range('2024-05','2025-03',freq='M').astype(str))
    monthly.to_csv(artifact('monthly_errors.csv'),index=False)
    # Refit solely to inspect the model; monthly errors above use saved predictions.
    rawx,rawz=build(a,aux,EXPERIMENTS['E3.3']),build(b,aux,EXPERIMENTS['E3.3'])
    model,x,z,pred,splits,seconds=fit_inspected(rawx,rawz,y,v,'ordinal')
    np.testing.assert_allclose(pred,saved['E3.3'],rtol=0,atol=1e-9)
    assert set(splits.decision_type)=={'<='}
    importance=pd.DataFrame({'feature':model.booster_.feature_name(),
                             'gain':model.booster_.feature_importance('gain'),
                             'split_count':model.booster_.feature_importance('split')})
    importance['gain_pct']=100*importance.gain/importance.gain.sum()
    importance['gain_rank']=importance.gain.rank(ascending=False,method='min').astype(int)
    importance.sort_values('gain',ascending=False).to_csv(artifact('e33_feature_importance.csv'),index=False)
    # Isolated sensitivity check only: no altered predictions are used for metrics.
    clip=z.copy(); clip['month_index']=x.month_index.max()
    clipped=model.predict(clip)
    low=z.copy();low['month_index']=z.month_index.min()
    high=z.copy();high['month_index']=z.month_index.max()
    audit_time=dict(train_month_index_range=[int(x.month_index.min()),int(x.month_index.max())],
                    validation_month_index_range=[int(z.month_index.min()),int(z.month_index.max())],
                    max_prediction_change_clipping_to_train_max=float(np.max(np.abs(pred-clipped))),
                    max_prediction_change_between_first_and_last_validation_month=float(np.max(np.abs(model.predict(low)-model.predict(high)))),
                    month_index_gain_rank=int(importance.set_index('feature').loc['month_index','gain_rank']),
                    month_index_gain_pct=float(importance.set_index('feature').loc['month_index','gain_pct']),
                    month_index_split_count=int(importance.set_index('feature').loc['month_index','split_count']),
                    refit_training_seconds=seconds,
                    saved_prediction_max_difference=float(np.max(np.abs(pred-saved['E3.3']))),
                    months_underpredicted=int(monthly['Mean Error'].lt(0).sum()),
                    months_total=len(monthly),
                    consecutive_rmse_increases=int(monthly.RMSE.diff().dropna().gt(0).sum()),
                    mean_error_linear_slope_sgd_per_month=float(np.polyfit(np.arange(len(monthly)),monthly['Mean Error'],1)[0]),
                    rmse_linear_slope_sgd_per_month=float(np.polyfit(np.arange(len(monthly)),monthly.RMSE,1)[0]),
                    trend_note='Slopes are unweighted descriptive summaries, not significance tests or drift tests.')
    write_json('temporal_audit.json',audit_time)
    # Training-only redundancy evidence, not a causal or incremental-importance estimate.
    numeric=build(a,aux,EXPERIMENTS['E3.4']).select_dtypes('number')
    numeric.corr(method='spearman').to_csv(artifact('training_numeric_spearman.csv'))
    monthly_macro=build(a,aux,EXPERIMENTS['E3.4'])[['month_index','coe_sum','stock_index']].drop_duplicates('month_index')
    monthly_macro.corr(method='spearman').to_csv(artifact('training_monthly_macro_spearman.csv'))
    ranges=[]
    aa,bb=build(a,aux,EXPERIMENTS['E3.4']),build(b,aux,EXPERIMENTS['E3.4'])
    for col in ['month_index','coe_sum','stock_index']:
        lo,hi=aa[col].min(),aa[col].max()
        ranges.append(dict(feature=col,train_unique=int(aa[col].nunique()),validation_unique=int(bb[col].nunique()),
                           train_min=lo,train_max=hi,validation_min=bb[col].min(),validation_max=bb[col].max(),
                           validation_outside_train_range_pct=float(100*((bb[col]<lo)|(bb[col]>hi)).mean())))
    pd.DataFrame(ranges).to_csv(artifact('temporal_feature_ranges.csv'),index=False)
    plots_temporal(monthly,importance)
    return monthly,importance,audit_time


def plots_temporal(monthly,importance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings=[('Mean Error','Monthly mean error (prediction − actual)','monthly_mean_error.png'),
              ('RMSE','Monthly prediction error','monthly_rmse.png')]
    for metric,title,file in settings:
        fig,ax=plt.subplots(figsize=(8,3.5))
        ax.plot(monthly.month,monthly[metric],marker='o',color='#487a9b')
        if metric=='Mean Error': ax.axhline(0,color='gray',linewidth=1)
        ax.set(title=title,ylabel=f'{metric} (SGD)');ax.tick_params(axis='x',rotation=45)
        ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(artifact(file),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.plot(monthly.month,monthly.actual_mean,marker='o',label='Actual mean')
    ax.plot(monthly.month,monthly.predicted_mean,marker='o',label='Predicted mean')
    ax.set(title='E3.3: Monthly actual and predicted mean rent',ylabel='Monthly rent (SGD)')
    ax.tick_params(axis='x',rotation=45);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(artifact('monthly_mean_rent.png'),dpi=180);plt.close(fig)
    top=importance.nlargest(12,'gain').sort_values('gain')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.barh(top.feature,top.gain_pct,color='#487a9b');ax.set(xlabel='Share of training split gain (%)',title='E3.3: Training gain importance')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(artifact('e33_gain_importance.png'),dpi=180);plt.close(fig)


def incremental_diagnostics():
    a,b,y,v,aux,audit=frames()
    prior_predictions(v)
    formal=pd.read_csv(OUT/'e3_results.csv').set_index('experiment')
    base=float(formal.loc['E3.3','RMSE'])
    rows=[];predictions={}
    for name in ['E3.3','E3.4']:
        row=formal.loc[name].to_dict();row['experiment']=name;row['origin']='reused official results';rows.append(row)
    for name,group in [('D1','malls'),('D2','coe'),('D3','stock')]:
        groups=EXPERIMENTS['E3.3']+(group,)
        aa,bb=build(a,aux,groups),build(b,aux,groups)
        assert list(aa)==CORE+[c for g in groups for c in GROUP_FEATURES[g]]
        row,pred=fit(name,aa,bb,y,v,'ordinal')
        row.update(feature_set='E3.3 + '+group.title(),auxiliary_groups=json.dumps(groups),
                   monthly_lag=1 if group in ['coe','stock'] else None,origin='new diagnostic fit')
        rows.append(row);predictions[name]=pred
    table=pd.DataFrame(rows)
    table['control']='E3.3';table['delta_rmse']=base-table.RMSE
    table['relative_improvement_pct']=100*table.delta_rmse/base
    table.to_csv(artifact('incremental_results.csv'),index=False)
    pd.DataFrame({'actual':v,**predictions},index=v.index).to_csv(artifact('incremental_predictions.csv'),index_label='listing_index')
    write_json('incremental_alignment_audit.json',dict(monthly_lag=audit['monthly_lag'],
               availability_assumption=audit['availability_assumption'],source_mapping=audit['monthly_source_mapping']))
    plot_incremental(table)
    return table


def plot_incremental(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delta = table[table.experiment.isin(['D1', 'D2', 'D3'])]
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.barh(delta.experiment, delta.delta_rmse,
            color=['#487a9b' if x >= 0 else '#ac6655' for x in delta.delta_rmse])
    ax.axvline(0, color='gray', linewidth=1)
    ax.set(xlabel='RMSE improvement vs E3.3 (SGD; positive is better)',
           title='Single auxiliary-group additions')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(artifact('incremental_rmse_changes.png'), dpi=180)
    plt.close(fig)
