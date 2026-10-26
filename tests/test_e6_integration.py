"""Check the E6 contrasts, future-time encoding, and paired loss accounting."""
import unittest
import numpy as np
import pandas as pd
from experiments.e6.run_e6 import CONFIGS, encode_config, analyze


class E6Tests(unittest.TestCase):
    def test_only_declared_groups_removed(self):
        self.assertEqual(set(CONFIGS['A'])-set(CONFIGS['B']), {'schools'})
        self.assertEqual(CONFIGS['C'],CONFIGS['B'])
        self.assertEqual(set(CONFIGS['B'])-set(CONFIGS['D']), {'mrt'})

    def test_calendar_is_onehot_and_future_year_is_not_clipped(self):
        a=pd.DataFrame({'FLOOR_AREA_SQM':[80.,90.], 'flat_age':[20.,21.],
                        'year':[2021,2022], 'calendar_month':['1','2'],
                        'TOWN':['a','b'], 'STREET':['road','road']},index=[5,7])
        b=pd.DataFrame({'FLOOR_AREA_SQM':[95.], 'flat_age':[22.],
                        'year':[2025], 'calendar_month':['2'],
                        'TOWN':['a'], 'STREET':['new']},index=[9])
        x,z=encode_config(a,b)
        self.assertNotIn('calendar_month',x)
        self.assertIn('calendar_month_2',x)
        self.assertEqual(z.loc[9,'calendar_month_2'],1)
        self.assertEqual(z.loc[9,'year'],2025)
        self.assertEqual(z.loc[9,'STREET'],-1)
        self.assertEqual(list(z.index),[9])

    def test_paired_sse_and_group_accounting(self):
        train=pd.DataFrame({'RENT_APPROVAL_DATE':['2022-01','2022-01'],
                            'TOWN':['a','b'],'BLOCK':['1','2'],'STREET':['x','y'],
                            'FLAT_TYPE':['3 room','4 room'],'FLAT_MODEL':['m','m'],
                            'FLOOR_AREA_SQM':[70.,100.],'LEASE_COMMENCE_DATE':[2000,2000],
                            'MONTHLY_RENT':[100.,200.]})
        valid=train.copy();valid.index=[5,7];valid.RENT_APPROVAL_DATE='2022-02'
        pred=pd.DataFrame({'actual':[100.,200.],'A':[90.,180.],'B':[100.,190.],
                           'C':[100.,190.],'D':[110.,200.]},index=valid.index)
        _,groups,pairs,_,_=analyze('test',train,valid,'2022-01',pred)
        pairs=pd.DataFrame(pairs)
        q=pairs[(pairs.candidate=='B')&(pairs.control=='A')&(pairs.dimension=='all')].iloc[0]
        self.assertEqual(q.sse_improvement,400.)
        self.assertEqual(q.mse_improvement,200.)
        self.assertEqual(q.fraction_rows_lower_absolute_error,1.)
        for _,g in pd.DataFrame(groups).groupby(['configuration','dimension']):
            self.assertEqual(g.rows.sum(),2)
            self.assertAlmostEqual(g.sse_share_pct.sum(),100.)


if __name__=='__main__': unittest.main()
