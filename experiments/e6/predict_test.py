"""Refit a fixed E6 configuration on all labeled rows and export Kaggle CSV.

Run: .venv/bin/python -m experiments.e6.predict_test --configuration C
The default is C (lowest formal E6 validation RMSE), not the parsimonious B choice.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.cleaning import clean_listings
from src.auxiliary import load_auxiliary
from src.features import make_auxiliary
from experiments.e6.run_e6 import ROOT, DEST, CONFIGS, PARAMS, make_config, encode_config, digest


def validate_submission(submission, sample, test):
    if list(submission.columns) != ['Id', 'Predicted']:
        raise ValueError('Expected Id,Predicted columns, without an extra index.')
    if len(submission) != len(test) or len(submission) != len(sample):
        raise ValueError('Submission row count mismatch.')
    if not submission.Id.equals(sample.Id) or not submission.Id.is_unique:
        raise ValueError('Submission IDs/order differ from sample.')
    if not np.isfinite(submission.Predicted).all():
        raise ValueError('Non-finite predictions.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',choices=list(CONFIGS),default='C')
    args=parser.parse_args(); name=args.configuration
    out=DEST/'submissions';out.mkdir(exist_ok=True)
    raw_train=pd.read_csv(ROOT/'data/train.csv')
    raw_test=pd.read_csv(ROOT/'data/test.csv')
    sample=pd.read_csv(ROOT/'data/example-submission.csv')
    assert 'MONTHLY_RENT' in raw_train and 'MONTHLY_RENT' not in raw_test
    assert list(sample.columns)==['Id','Predicted']
    assert sample.Id.tolist()==list(range(len(raw_test))), 'Confirm ID-to-test-row mapping.'
    train=clean_listings(raw_train,drop_exact_duplicates=True)
    test=clean_listings(raw_test,drop_exact_duplicates=False)
    assert test.index.equals(raw_test.index) and len(test)==len(raw_test)
    assert train.MONTHLY_RENT.notna().all()
    raw=load_auxiliary()
    aux=make_auxiliary(df_hdb=raw['hdb'],df_mrt=raw['mrt'],df_schools=raw['schools'],df_malls=raw['malls'])
    a,b=make_config(train,aux,name),make_config(test,aux,name)
    x,z=encode_config(a,b)
    model=LGBMRegressor(**PARAMS).fit(x,train.MONTHLY_RENT)
    predictions=model.predict(z)
    submission=pd.DataFrame({'Id':sample.Id.copy(),'Predicted':predictions})
    validate_submission(submission,sample,test)
    path=out/f'submission_e6_{name}_fulltrain.csv'
    submission.to_csv(path,index=False)
    reread=pd.read_csv(path);validate_submission(reread,sample,test)
    np.testing.assert_allclose(reread.Predicted,predictions,rtol=0,atol=1e-9)
    monthly=pd.DataFrame({'month':test.RENT_APPROVAL_DATE,'predicted':predictions}).groupby('month').predicted.agg(['size','mean','std','min','max'])
    monthly.to_csv(out/f'e6_{name}_test_monthly_predictions.csv')
    audit=dict(configuration=name,selection='Lowest original formal E6 RMSE' if name=='C' else 'Explicitly requested configuration',
               caution='C was worse in historical E6 backtests; this is a first submission candidate, not a demonstrated robust winner.',
               train_rows=len(train),raw_train_rows=len(raw_train),test_rows=len(test),
               train_period=[train.RENT_APPROVAL_DATE.min(),train.RENT_APPROVAL_DATE.max()],
               test_period=[test.RENT_APPROVAL_DATE.min(),test.RENT_APPROVAL_DATE.max()],
               includes_former_validation_in_training=True,test_deduplicated=False,
               raw_features=list(a),model_params=PARAMS,
               encoding_fit='all labeled training rows only; no joint train/test category fitting',
               target_transform='none',prediction_rounding_or_clipping='none',
               prediction_summary=pd.Series(predictions).describe().to_dict(),
               checks={'rows_and_ids_match_sample':True,'all_predictions_finite':True,'test_order_preserved':True},
               versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','scikit-learn']},
               input_hashes={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'data/train.csv',ROOT/'data/test.csv',ROOT/'data/example-submission.csv',*sorted((ROOT/'data/auxiliary').glob('*.csv'))]},
               source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),DEST/'run_e6.py',ROOT/'src/features.py',ROOT/'src/cleaning.py',ROOT/'src/auxiliary.py']},
               submission_sha256=digest(path),uploaded=False,kaggle_score=None)
    (out/f'e6_{name}_submission_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'file':str(path),'train_rows':len(train),'test_rows':len(test),'prediction_summary':audit['prediction_summary']},indent=2),flush=True)


if __name__=='__main__':main()
