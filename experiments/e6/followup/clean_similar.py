"""Training-only peer outlier filter. Never apply to evaluation labels.

Fixed rule: quarter/town/flat type/20sqm bin, fallback to quarter/town/type,
then quarter/type. Minimum 30 peers. Flag log rent outside median +/-
max(3.5 * 1.4826 * MAD, log(1.5)). No validation-driven threshold tuning.
"""
import numpy as np
import pandas as pd


def flag_training_outliers(train):
    work = train.copy()
    work['quarter'] = pd.to_datetime(work.RENT_APPROVAL_DATE).dt.to_period('Q').astype(str)
    work['area_bin'] = (work.FLOOR_AREA_SQM // 20).astype('Int64')
    work['log_rent'] = np.log(work.MONTHLY_RENT.where(work.MONTHLY_RENT > 0))
    audit = pd.DataFrame(index=train.index, columns=['peer_level','peer_count','peer_median_log','peer_mad_log'])
    for keys in [['quarter','TOWN','FLAT_TYPE','area_bin'], ['quarter','TOWN','FLAT_TYPE'], ['quarter','FLAT_TYPE']]:
        groups=work.groupby(keys, dropna=False)['log_rent']
        n=groups.transform('count'); med=groups.transform('median')
        deviation=(work.log_rent-med).abs()
        mad=deviation.groupby([work[k] for k in keys],dropna=False).transform('median')
        use=audit.peer_level.isna() & n.ge(30)
        audit.loc[use,'peer_level']='+'.join(keys)
        audit.loc[use,'peer_count']=n[use]
        audit.loc[use,'peer_median_log']=med[use]
        audit.loc[use,'peer_mad_log']=mad[use]
    width=np.maximum(3.5*1.4826*audit.peer_mad_log.astype(float),np.log(1.5))
    audit['lower_rent']=np.exp(audit.peer_median_log.astype(float)-width)
    audit['upper_rent']=np.exp(audit.peer_median_log.astype(float)+width)
    audit['remove']=train.MONTHLY_RENT.le(0) | train.MONTHLY_RENT.lt(audit.lower_rent) | train.MONTHLY_RENT.gt(audit.upper_rent)
    audit['rent']=train.MONTHLY_RENT
    return audit
