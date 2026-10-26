"""E5 guards against temporal leakage and silent prediction misalignment."""
import unittest
import numpy as np
import pandas as pd
from experiments.error_analysis.e5 import split_at, validate_predictions, profile


class E5Tests(unittest.TestCase):
    def test_split_is_fixed_origin_and_requires_full_horizon(self):
        frame = pd.DataFrame({'RENT_APPROVAL_DATE': ['2022-01','2022-02','2022-03','2022-04']}, index=[9,3,7,1])
        train, valid = split_at(frame, '2022-02', 2)
        self.assertEqual(list(train.index), [9,3])
        self.assertEqual(list(valid.index), [7,1])
        with self.assertRaises(ValueError):
            split_at(frame, '2022-02', 3)

    def test_shuffled_predictions_are_rejected_even_when_labels_match(self):
        valid = pd.DataFrame({'MONTHLY_RENT': [100.,100.]}, index=[3,7])
        pred = pd.DataFrame({'actual': [100.,100.], 'model': [90.,110.]}, index=[7,3])
        with self.assertRaises(ValueError):
            validate_predictions(pred, valid)
        validate_predictions(pred.loc[valid.index], valid)

    def test_support_and_price_edges_do_not_learn_validation_labels(self):
        train = pd.DataFrame({'RENT_APPROVAL_DATE': ['2022-01']*2, 'TOWN': ['a']*2,
                              'BLOCK': ['1']*2, 'STREET': ['road']*2, 'FLAT_TYPE': ['4 room']*2,
                              'FLAT_MODEL': ['model']*2, 'FLOOR_AREA_SQM': [90.,90.],
                              'LEASE_COMMENCE_DATE': [2000,2000], 'MONTHLY_RENT': [100.,200.]})
        valid = train.iloc[[0]].copy()
        valid.index = [9]
        valid.RENT_APPROVAL_DATE = '2022-02'
        valid.BLOCK = 'unknown'
        before, edges = profile(train, valid, '2022-01')
        valid.MONTHLY_RENT = 999999.
        after, new_edges = profile(train, valid, '2022-01')
        self.assertEqual(edges, new_edges)
        self.assertEqual(before.loc[9, 'address_support'], 'unseen')
        self.assertEqual(after.loc[9, 'horizon'], 1)
        self.assertEqual(before.loc[9, 'address_support'], after.loc[9, 'address_support'])


if __name__ == '__main__':
    unittest.main()
