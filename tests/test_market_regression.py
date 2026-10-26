import unittest
import pandas as pd
import numpy as np
from experiments.e6.market_regression.run import columns, extrapolate, temporal_targets
from experiments.e6.market_regression.catboost_trend import residual_features

class MarketRegressionTests(unittest.TestCase):
    def test_extrapolation_and_damping(self):
        effects=pd.Series(np.arange(12)*10.,index=np.arange(12))
        counts=pd.Series(100,index=effects.index)
        query=np.array([12,17,1000])
        linear,_=extrapolate(effects,counts,query,'linear')
        flat,_=extrapolate(effects,counts,query,'flat')
        damped,_=extrapolate(effects,counts,query,'damped')
        np.testing.assert_allclose(linear,query*10,atol=1e-8)
        np.testing.assert_allclose(flat,110)
        self.assertTrue(np.all(damped>=flat))
        self.assertTrue(np.all(damped<=linear+1e-8))
        self.assertLessEqual(damped[-1],170+1e-8)
    def test_target_encoding_no_current_or_future_month(self):
        a=pd.DataFrame({'TOWN':['a']*4,'FLAT_TYPE':['5 room']*4,'calendar_month':['1','2','2','3'],'STREET':['s']*4,'composite_block_id':['b']*4})
        b=a.iloc[[0]].copy();b.index=[5]
        y=pd.Series([1000.,2000.,4000.,9000.])
        ma=pd.Series([1,2,2,3]);mb=pd.Series([4],index=[5])
        x,z=temporal_targets(a,b,y,ma,mb)
        self.assertTrue(np.isnan(x.loc[0,'STREET']))
        self.assertEqual(x.loc[1,'STREET'],1000.)
        self.assertEqual(x.loc[2,'STREET'],1000.)
        yy=y.copy();yy.loc[1:]=99999
        xx,_=temporal_targets(a,b,yy,ma,mb)
        np.testing.assert_allclose(x.STREET.iloc[:3],xx.STREET.iloc[:3],equal_nan=True)
        self.assertAlmostEqual(z.loc[5,'STREET'],y.mean())
    def test_unknown_target_category_uses_training_prior(self):
        a=pd.DataFrame({'calendar_month':['1','2'],'STREET':['s','s'],'composite_block_id':['b','b']})
        b=pd.DataFrame({'calendar_month':['2'],'STREET':['unknown'],'composite_block_id':['unknown']},index=[8])
        _,z=temporal_targets(a,b,pd.Series([100.,300.]),pd.Series([1,2]),pd.Series([3],index=[8]))
        self.assertEqual(z.loc[8,'STREET'],200.)
        self.assertEqual(z.loc[8,'STREET_count'],0.)

    def test_category_columns_support_b_and_c_time_representations(self):
        base=pd.DataFrame({'TOWN':['a'],'FLAT_TYPE':['5 room'],
                           'STREET':['s'],'composite_block_id':['b']})
        low_b,high_b=columns(base.assign(month_index=1))
        low_c,high_c=columns(base.assign(year=2024,calendar_month='1'))
        self.assertNotIn('calendar_month',low_b)
        self.assertIn('calendar_month',low_c)
        self.assertIn('composite_block_id',high_b)
        self.assertIn('composite_block_id',high_c)

    def test_catboost_trend_residual_has_only_seasonal_time(self):
        frame=pd.DataFrame({'month_index':[40], 'year':[2024],
                            'calendar_month':['5'], 'FLOOR_AREA_SQM':[90]})
        result=residual_features(frame)
        self.assertNotIn('month_index',result)
        self.assertNotIn('year',result)
        self.assertIn('calendar_month',result)
        self.assertIn('FLOOR_AREA_SQM',result)

if __name__=='__main__':unittest.main()
