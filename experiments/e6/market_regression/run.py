"""Future cross-sectional regression: encoding and market-trend experiments.
Run: .venv/bin/python -m experiments.e6.market_regression.run
Fixed choices before evaluating: no search or tuning on outer validation.
"""
import json
import time
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from src.validation import load_data
from src.auxiliary import load_auxiliary
from src.features import make_auxiliary, feature_groups, onehot_encoding
from experiments.e6.run_e6 import make_config, score, digest
from experiments.e6.followup.run import encode
from experiments.feature_engineering.utils.core import ROOT, PARAMS, composite
from experiments.error_analysis.e5 import FOLDS, split_at

OUT=Path(__file__).resolve().parent/'results'
CAT_PARAMS=dict(iterations=1000,depth=6,learning_rate=.05,loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False,has_time=True)


def address_features(a,aux,config='C'):
    x=make_config(a,aux,config).drop(columns='BLOCK')
    x['composite_block_id']=composite(a)
    return x


def columns(x):
    _,low,high=feature_groups(x)
    if 'calendar_month' in x:
        low=low+['calendar_month']
    return low,high+['composite_block_id']


def fit_catboost(train, valid, y):
    low,high=columns(train);cats=low+high
    x=train.copy();z=valid.copy()
    for col in cats:
        x[col]=x[col].fillna('MISSING').astype(str)
        z[col]=z[col].fillna('MISSING').astype(str)
    for col in x.columns.difference(cats):
        median=x[col].median()
        x[col]=x[col].fillna(median)
        z[col]=z[col].fillna(median)
    if 'month_index' in x:
        chronology=x['month_index']
    else:
        chronology=x['year']*12+x['calendar_month'].astype(int)
    order=chronology.sort_values(kind='stable').index
    model=CatBoostRegressor(**CAT_PARAMS).fit(x.loc[order],y.loc[order],cat_features=cats)
    train_rmse=float(model.get_evals_result()['learn']['RMSE'][-1])
    return model.predict(z),train_rmse


def sparse_encoder(x):
    low,high=columns(x); cats=low+high
    numeric=[c for c in x if c not in cats]
    return ColumnTransformer([
        ('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric),
        ('categorical',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=10,sparse_output=True),cats)
    ],sparse_threshold=1.)


def temporal_targets(a,b,y,months_a,months_b):
    """Replace high-card labels with expanding, prior-month smoothed means.
    20 pseudo-observations at prior-month global mean. Add category counts.
    Validation receives only training-cutoff aggregates. Early train => NaN.
    """
    low,high=columns(a)
    x,z=onehot_encoding(a,b,columns=low)
    for frame in [x,z]:
        for col in high:
            frame[col]=np.nan;frame[col+'_count']=0.
    for month in sorted(months_a.unique()):
        idx=months_a.index[months_a.eq(month)]
        past=months_a.index[months_a.lt(month)]
        prior=y.loc[past].mean()
        for col in high:
            stats=pd.DataFrame({'key':a.loc[past,col],'y':y.loc[past]}).groupby('key').y.agg(['sum','count'])
            count=a.loc[idx,col].map(stats['count']).fillna(0)
            total=a.loc[idx,col].map(stats['sum']).fillna(0)
            x.loc[idx,col]=(total+20*prior)/(count+20)
            x.loc[idx,col+'_count']=count
    assert months_b.min()>months_a.max()
    for col in high:
        stats=pd.DataFrame({'key':a[col],'y':y}).groupby('key').y.agg(['sum','count'])
        count=b[col].map(stats['count']).fillna(0)
        z[col]=(b[col].map(stats['sum']).fillna(0)+20*y.mean())/(count+20)
        z[col+'_count']=count
    return x,z


def extrapolate(effects, counts, query, mode):
    """Regression on last12 adjusted month effects, anchored to last3 average.
    Mode flat isolates detrending; linear extrapolates; damped caps accumulated
    future slope at six month-equivalents. These settings are fixed, not tuned.
    """
    tail=effects.iloc[-12:]; t=tail.index.to_numpy(dtype=float)
    slope=np.polyfit(t-t.max(),tail.to_numpy(),1,w=np.sqrt(counts.loc[tail.index]))[0]
    anchor=effects.iloc[-3:].mean();anchor_t=np.mean(effects.index[-3:])
    cutoff=int(effects.index.max())
    at_cut=anchor+slope*(cutoff-anchor_t)
    h=np.maximum(np.asarray(query,dtype=float)-cutoff,0)
    advance={'flat':np.zeros_like(h),'linear':h,'damped':6*(-np.expm1(-h/6))}[mode]
    return at_cut+slope*advance,dict(slope_per_month=float(slope),cutoff_effect=float(at_cut),mode=mode)


def fit_market(x,z,y,ma,mb):
    # Joint hedonic ridge: month fixed effects plus observed housing attributes.
    # Address describes a group of flats, never a longitudinal individual ID.
    xx=x.drop(columns=['year','calendar_month']);zz=z.drop(columns=['year','calendar_month'])
    encoder=sparse_encoder_no_time(xx)
    u=encoder.fit_transform(xx);v=encoder.transform(zz)
    month_encoder=OneHotEncoder(sparse_output=True,handle_unknown='ignore')
    m=month_encoder.fit_transform(ma.to_numpy().reshape(-1,1))
    # Use an explicit intercept column.  This avoids the sparse Ridge
    # intercept reconstruction warning and keeps the month coefficients
    # numerically stable enough for slope estimation.
    constant=sparse.csr_matrix(np.ones((len(y),1)))
    design=sparse.hstack([constant,u,m],format='csr')
    ridge=Ridge(alpha=20,solver='lsqr',tol=1e-6,fit_intercept=False).fit(design,y)
    coefficients=ridge.coef_[-m.shape[1]:]
    effects=pd.Series(coefficients,index=month_encoder.categories_[0].astype(int))
    train_effect=ma.map(effects)
    property_valid=(ridge.coef_[0]
                    + np.asarray(v @ ridge.coef_[1:1+u.shape[1]]).ravel())
    return train_effect,effects,ma.value_counts(),property_valid


def sparse_encoder_no_time(x):
    cats=list(x.select_dtypes(include=['object','string']).columns)
    num=[c for c in x if c not in cats]
    return ColumnTransformer([
        ('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),
        ('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=10,sparse_output=True),cats)
    ],sparse_threshold=1.)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train,valid=load_data();raw=load_auxiliary()
    aux=make_auxiliary(df_hdb=raw['hdb'],df_mrt=raw['mrt'],df_malls=raw['malls'])
    folds=[('formal_2024_04',train,valid)]+[(n,*split_at(train,c,h)) for n,c,h in FOLDS]
    rows=[];groups=[];trend_rows=[]
    protocol={'data_sha256':digest(ROOT/'data/train.csv'),'script_sha256':digest(Path(__file__)),
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','catboost']},
              'lgbm':PARAMS,'catboost':CAT_PARAMS,'cleaning':'original shared cleaning only; no new outlier exclusion',
              'address':'town/street/block group, not unique flat','sparse_onehot_min_frequency':10,'target_prior_weight':20,
              'trend':{'hedonic_ridge_alpha':20,'recent_slope_months':12,'anchor_months':3,'damping_months':6},
              'selection':'all choices fixed before outer validation; exploratory comparisons; no test.csv or outer labels in fitting'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for fold,a,b in folds:
        x=address_features(a,aux);z=address_features(b,aux);y=a.MONTHLY_RENT
        ma=pd.Series(pd.PeriodIndex(a.RENT_APPROVAL_DATE,freq='M').asi8,index=a.index)
        mb=pd.Series(pd.PeriodIndex(b.RENT_APPROVAL_DATE,freq='M').asi8,index=b.index)
        preds=pd.DataFrame({'actual':b.MONTHLY_RENT})
        def save(name,p,seconds=0,train_rmse=np.nan):
            p=np.asarray(p);assert np.isfinite(p).all() and len(p)==len(b)
            preds[name]=p
            row=dict(fold=fold,configuration=name,train_rows=len(a),validation_rows=len(b),seconds=seconds,train_rmse=train_rmse,**score(b.MONTHLY_RENT,p))
            rows.append(row);print(json.dumps(row),flush=True)
            errors=preds[name]-preds.actual
            for dim,labels in [('month',b.RENT_APPROVAL_DATE),('rent_band',pd.cut(b.MONTHLY_RENT,[-np.inf,2000,3600,np.inf]).astype(str))]:
                for label,idx in labels.groupby(labels).groups.items():
                    groups.append(dict(fold=fold,configuration=name,dimension=dim,group=str(label),rows=len(idx),sse_share=float((errors.loc[idx]**2).sum()/(errors**2).sum()),**score(preds.loc[idx,'actual'],preds.loc[idx,name])))
            pd.DataFrame(rows).to_csv(OUT/'overall.csv',index=False)
            pd.DataFrame(groups).to_csv(OUT/'group_errors.csv',index=False)
            preds.to_csv(OUT/f'{fold}_predictions.csv.gz',index_label='listing_index')
        old=pd.read_csv(ROOT/'experiments/e6/followup/results'/f'{fold}_predictions.csv.gz',index_col='listing_index')
        np.testing.assert_allclose(old.loc[b.index,'actual'],b.MONTHLY_RENT)
        for config in ['C_baseline','C_address','C_address_native','C_clean']:
            save(config,old.loc[b.index,config])
        orig=pd.read_csv(ROOT/'experiments/e6/predictions'/f'{fold}.csv.gz',index_col='listing_index')
        np.testing.assert_allclose(orig.loc[b.index,'actual'],b.MONTHLY_RENT)
        save('B_month_index',orig.loc[b.index,'B'])
        for method in ['frequency','onehot','temporal_target']:
            start=time.perf_counter()
            if method=='frequency': u,v=encode(x,z,'frequency')
            elif method=='onehot':
                enc=sparse_encoder(x);u=enc.fit_transform(x);v=enc.transform(z)
            else:u,v=temporal_targets(x,z,y,ma,mb)
            model=LGBMRegressor(**PARAMS).fit(u,y)
            save('C_address_'+method,model.predict(v),time.perf_counter()-start,score(y,model.predict(u))['RMSE'])
        start=time.perf_counter()
        cat_pred,cat_train_rmse=fit_catboost(x,z,y)
        save('C_address_catboost',cat_pred,time.perf_counter()-start,cat_train_rmse)
        start=time.perf_counter()
        bx=address_features(a,aux,'B');bz=address_features(b,aux,'B')
        cat_pred,cat_train_rmse=fit_catboost(bx,bz,y)
        save('B_address_catboost',cat_pred,time.perf_counter()-start,cat_train_rmse)
        start=time.perf_counter()
        train_effect,effects,counts,ridge_property=fit_market(x,z,y,ma,mb)
        effects.to_csv(OUT/f'{fold}_market_effects.csv',header=['effect'],index_label='month_ordinal')
        u,v=encode(x,z,'ordinal')
        residual_model=LGBMRegressor(**PARAMS).fit(u,y-train_effect)
        residual_prediction=residual_model.predict(v)
        for mode in ['flat','linear','damped']:
            future,detail=extrapolate(effects,counts,mb,mode)
            trend_rows.append(dict(fold=fold,**detail))
            save('Trend_'+mode+'_LGBM',future+residual_prediction,time.perf_counter()-start,score(y,train_effect+residual_model.predict(u))['RMSE'])
            if mode=='damped':save('Hedonic_Ridge_damped',future+ridge_property)
        pd.DataFrame(trend_rows).to_csv(OUT/'trend_parameters.csv',index=False)
    print('Complete',flush=True)

if __name__=='__main__':main()
