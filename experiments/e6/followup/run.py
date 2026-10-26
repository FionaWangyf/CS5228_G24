"""Run with: .venv/bin/python -m experiments.e6.followup.run
Independent ablations; fixed original E6-C model parameters, no test.csv.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from src.validation import load_data
from src.auxiliary import load_auxiliary
from src.features import make_auxiliary, feature_groups, onehot_encoding, encode_high_card
from experiments.e6.run_e6 import make_config, score, digest
from experiments.feature_engineering.utils.core import PARAMS, composite, ROOT
from experiments.error_analysis.e5 import FOLDS, split_at
from .clean_similar import flag_training_outliers

OUT=Path(__file__).resolve().parent/'results'
SPECS={
'C_baseline': (False,False,'ordinal',False),
'C_clean': (True,False,'ordinal',False),
'C_address': (False,True,'ordinal',False),
'C_address_native': (False,True,'native',False),
'C_address_history': (False,True,'ordinal',True),
'C_address_native_history': (False,True,'native',True),
'C_clean_address_native_history': (True,True,'native',True),
}


def history_features(train, target):
    """Strict earlier-month, trailing six-month statistics; freeze at fit cutoff.
    Hierarchical means: global -> town/type -> street/type -> building/type.
    Prior weight 20, raw counts retained. No history => NaN mean, zero count.
    """
    source=train.copy(); query=target.copy()
    source['_month']=pd.PeriodIndex(source.RENT_APPROVAL_DATE,freq='M').asi8
    query['_month']=pd.PeriodIndex(query.RENT_APPROVAL_DATE,freq='M').asi8
    source['_building']=composite(source); query['_building']=composite(query)
    cutoff=int(source._month.max())
    result=pd.DataFrame(index=query.index)
    levels=[('town',['TOWN','FLAT_TYPE']),('street',['TOWN','STREET','FLAT_TYPE']),('building',['_building','FLAT_TYPE'])]
    for month, batch in query.groupby('_month'):
        end=min(int(month)-1,cutoff)
        past=source[source._month.between(end-5,end)]
        prior=pd.Series(past.MONTHLY_RENT.mean(),index=batch.index)
        for name,keys in levels:
            ag=past.groupby(keys).MONTHLY_RENT.agg(['sum','count'])
            lookup=pd.MultiIndex.from_frame(batch[keys])
            vals=ag.reindex(lookup)
            counts=pd.Series(vals['count'].fillna(0).to_numpy(),index=batch.index)
            sums=pd.Series(vals['sum'].fillna(0).to_numpy(),index=batch.index)
            mean=(sums+20*prior)/(counts+20)
            result.loc[batch.index,f'hist_{name}_mean']=mean
            result.loc[batch.index,f'hist_{name}_count']=counts
            prior=mean
    return result


def encode(a,b,method):
    _,low,high=feature_groups(a)
    low+=['calendar_month']
    if 'composite_block_id' in a: high+=['composite_block_id']
    x,z=onehot_encoding(a,b,columns=low)
    return encode_high_card(x,z,method=method,columns=high)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train,valid=load_data(); raw=load_auxiliary()
    aux=make_auxiliary(df_hdb=raw['hdb'],df_mrt=raw['mrt'],df_malls=raw['malls'])
    folds=[('formal_2024_04',train,valid)]+[(name,*split_at(train,cut,h)) for name,cut,h in FOLDS]
    rows=[]; groups=[]
    (OUT/'protocol.json').write_text(json.dumps({'specs':SPECS,'params':PARAMS,'filter':'quarter/town/type/area peers; min30; log median +/- max(3.5*1.4826*MAD,log1.5)','history':'strict prior month trailing6, hierarchical smoothing weight20; freeze validation at train cutoff','test_used':False,'validation_rows_removed':0,'train_sha256':digest(ROOT/'data/train.csv'),'source_hashes':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}},indent=2))
    for fold,a,b in folds:
        flags=flag_training_outliers(a); flags.to_csv(OUT/f'{fold}_cleaning.csv',index_label='listing_index')
        base_a=make_config(a,aux,'C'); base_b=make_config(b,aux,'C')
        preds=pd.DataFrame({'actual':b.MONTHLY_RENT})
        histories={}
        for name,(clean,address,method,hist) in SPECS.items():
            start=time.perf_counter(); fit=a.loc[~flags.remove] if clean else a
            xraw=base_a.loc[fit.index].copy(); zraw=base_b.copy()
            if address:
                xraw=xraw.drop(columns='BLOCK'); zraw=zraw.drop(columns='BLOCK')
                xraw['composite_block_id']=composite(fit);zraw['composite_block_id']=composite(b)
            if hist:
                if clean not in histories: histories[clean]=(history_features(fit,fit),history_features(fit,b))
                hx,hz=histories[clean];xraw=xraw.join(hx);zraw=zraw.join(hz)
            x,z=encode(xraw,zraw,method)
            assert x.index.equals(fit.index) and z.index.equals(b.index)
            cats=[c for c in x if isinstance(x[c].dtype,pd.CategoricalDtype)]
            model=LGBMRegressor(**PARAMS).fit(x,fit.MONTHLY_RENT,categorical_feature=cats or 'auto')
            p=model.predict(z);assert np.isfinite(p).all();preds[name]=p
            if name=='C_baseline':
                prior=pd.read_csv(ROOT/'experiments/e6/predictions'/f'{fold}.csv.gz',index_col='listing_index')
                np.testing.assert_allclose(p,prior.loc[b.index,'C'],atol=1e-8,rtol=0)
            row=dict(fold=fold,configuration=name,train_rows=len(fit),removed=len(a)-len(fit),validation_rows=len(b),seconds=time.perf_counter()-start,train_rmse=score(fit.MONTHLY_RENT,model.predict(x))['RMSE'],**score(b.MONTHLY_RENT,p))
            rows.append(row); print(json.dumps(row),flush=True)
            err=p-b.MONTHLY_RENT
            for dim,labels in [('month',b.RENT_APPROVAL_DATE),('rent_band',pd.cut(b.MONTHLY_RENT,[-np.inf,2000,3600,np.inf]).astype(str)),('town',b.TOWN)]:
                for label,idx in labels.groupby(labels).groups.items():
                    groups.append(dict(fold=fold,configuration=name,dimension=dim,group=str(label),rows=len(idx),sse_share=float((err.loc[idx]**2).sum()/(err**2).sum()),**score(b.loc[idx,'MONTHLY_RENT'],preds.loc[idx,name])))
            pd.DataFrame(rows).to_csv(OUT/'overall.csv',index=False)
            pd.DataFrame(groups).to_csv(OUT/'group_errors.csv',index=False)
            preds.to_csv(OUT/f'{fold}_predictions.csv.gz',index_label='listing_index')
    print('Complete',flush=True)

if __name__=='__main__':main()
