import unittest
import pandas as pd
import numpy as np
from experiments.e6.followup.run import history_features
from experiments.e6.followup.clean_similar import flag_training_outliers

class FollowupTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'RENT_APPROVAL_DATE':['2024-01','2024-02','2024-02','2024-03'], 'TOWN':['a']*4,'STREET':['s']*4,'BLOCK':['1']*4,'FLAT_TYPE':['5 room']*4,'FLOOR_AREA_SQM':[110]*4,'MONTHLY_RENT':[1000,2000,4000,9000]})
    def test_history_excludes_current_and_future_labels(self):
        a=self.frame();h=history_features(a,a)
        self.assertTrue(np.isnan(h.loc[0,'hist_town_mean']))
        self.assertEqual(h.loc[1,'hist_town_mean'],1000)
        self.assertEqual(h.loc[2,'hist_town_mean'],1000)
        changed=a.copy();changed.loc[1:,'MONTHLY_RENT']=99999
        hh=history_features(changed,changed)
        pd.testing.assert_frame_equal(h.iloc[:3],hh.iloc[:3])
    def test_future_history_frozen_and_target_ignored(self):
        a=self.frame();q=a.iloc[[0,1]].copy();q.RENT_APPROVAL_DATE=['2024-04','2025-08']
        h=history_features(a,q);q.MONTHLY_RENT=-999
        pd.testing.assert_frame_equal(h,history_features(a,q))
        np.testing.assert_allclose(h.iloc[0],h.iloc[1])
    def test_peer_filter_low_high_and_sparse(self):
        a=pd.concat([self.frame().iloc[[0]]]*40,ignore_index=True);a.MONTHLY_RENT=3000
        a.loc[0,'MONTHLY_RENT']=1000;a.loc[1,'MONTHLY_RENT']=7000
        flags=flag_training_outliers(a)
        self.assertEqual(flags.index[flags.remove].tolist(),[0,1])
        self.assertFalse(flag_training_outliers(a.iloc[:3]).remove.any())

if __name__=='__main__':unittest.main()
