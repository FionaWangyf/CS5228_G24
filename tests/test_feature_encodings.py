"""Behavioral checks for train-only high-card encoding and composite IDs."""
import unittest
import numpy as np
import pandas as pd
from src.features import encode_high_card
from experiments.feature_engineering.utils.core import composite

class EncodingTests(unittest.TestCase):
    def test_train_only_unknown_and_index(self):
        a=pd.DataFrame({'BLOCK':['a','a','b',None], 'value':[1,np.nan,3,4]}, index=[8,2,5,9])
        b=pd.DataFrame({'BLOCK':['a','new',None], 'value':[1,np.nan,3]}, index=[7,1,6])
        for method in ['ordinal','frequency','native']:
            x,z=encode_high_card(a,b,method=method)
            self.assertTrue(x.index.equals(a.index))
            self.assertTrue(z.index.equals(b.index))
            pd.testing.assert_series_equal(x.value,a.value)
            if method=='ordinal':
                self.assertEqual(z.loc[1,'BLOCK'],-1)
                self.assertTrue(pd.api.types.is_numeric_dtype(x.BLOCK))
            elif method=='frequency':
                np.testing.assert_allclose(z.BLOCK,[.5,0,.25])
            else:
                self.assertIsInstance(x.BLOCK.dtype,pd.CategoricalDtype)
                self.assertTrue(pd.isna(z.loc[1,'BLOCK']))
                self.assertEqual(list(x.BLOCK.cat.categories),list(z.BLOCK.cat.categories))
                self.assertNotIn('new',x.BLOCK.cat.categories)
        # Validation composition cannot alter train encodings.
        for method in ['ordinal','frequency','native']:
            x,_=encode_high_card(a,b,method=method)
            xx,_=encode_high_card(a,b.iloc[[1]],method=method)
            pd.testing.assert_frame_equal(x,xx)

    def test_composite_no_separator_collision(self):
        x=pd.DataFrame({'TOWN':['a|b','a','x','x'], 'BLOCK':['c','b|c',None,'null'], 'STREET':['d','d','y','y']})
        self.assertEqual(composite(x).nunique(),4)

if __name__=='__main__': unittest.main()
