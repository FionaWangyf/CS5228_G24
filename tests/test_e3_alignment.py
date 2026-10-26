"""Calendar lag behavior and strict E3 group inclusion."""
import unittest
import pandas as pd
from experiments.feature_engineering.utils.auxiliary import lag_monthly_auxiliary, build, CORE


class E3AlignmentTests(unittest.TestCase):
    def test_calendar_gap_and_start_remain_missing(self):
        coe = pd.DataFrame({'month_index': [0, 2], 'coe_sum': [10., 30.]})
        original = coe.copy(deep=True)
        lagged = lag_monthly_auxiliary({'coe': coe})
        pd.testing.assert_frame_equal(coe, original)
        listing = pd.DataFrame({'month_index': [0, 1, 2, 3]}, index=[6, 8, 3, 2])
        joined = listing.join(lagged['coe'].set_index('month_index'), on='month_index')
        self.assertTrue(joined.index.equals(listing.index))
        self.assertTrue(pd.isna(joined.loc[6, 'coe_sum']))
        self.assertEqual(joined.loc[8, 'coe_sum'], 10.)
        self.assertTrue(pd.isna(joined.loc[3, 'coe_sum']))
        self.assertEqual(joined.loc[2, 'coe_sum'], 30.)

    def test_future_source_cannot_change_earlier_join(self):
        macro = pd.DataFrame({'month_index': [0, 1, 2], 'stock_index': [1., 2., 3.]})
        modified = macro.copy()
        modified.loc[2, 'stock_index'] = 999999.
        a = lag_monthly_auxiliary({'stock': macro})['stock'].set_index('month_index')
        b = lag_monthly_auxiliary({'stock': modified})['stock'].set_index('month_index')
        pd.testing.assert_frame_equal(a.loc[:2], b.loc[:2])
        with self.assertRaises(ValueError): lag_monthly_auxiliary({'stock': macro}, months=0)
        with self.assertRaises(ValueError):
            lag_monthly_auxiliary({'stock': pd.concat([macro, macro.iloc[[0]]])})

    def test_hdb_only_excludes_other_groups(self):
        core = pd.DataFrame([[90., 20., 5, 'town', '4 room', 'model a', '1', 'road']], columns=CORE, index=[17])
        hdb = pd.DataFrame({'TOWN': ['town'], 'BLOCK': ['1'], 'STREET': ['road'],
                            'LATITUDE': [1.3], 'LONGITUDE': [103.8], 'MAX_FLOOR': [12],
                            'SUBZONE': ['subzone'], 'PLANNING_AREA': ['area']})
        auxiliary = {'hdb': hdb, 'coe': pd.DataFrame({'month_index': [5], 'coe_sum': [100.]})}
        result = build(core, auxiliary, ('hdb',))
        self.assertEqual(result.shape, (1, 13))
        self.assertNotIn('coe_sum', result)
        pd.testing.assert_frame_equal(build(core, auxiliary, ()), core)


if __name__ == '__main__': unittest.main()
