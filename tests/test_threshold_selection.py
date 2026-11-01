import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import unittest
import numpy as np
from select_threshold import choose_threshold, metrics


class ThresholdTests(unittest.TestCase):
    def test_recall_constraint(self):
        y=np.array([1,1,1,0,0]);s=np.array([.9,.8,.2,.7,.1])
        t=choose_threshold(y,s,.6)
        self.assertEqual(t,.8)
        self.assertEqual(metrics(y,s>=t)['tp'],2)
        self.assertEqual(metrics(y,s>=t)['fp'],0)

    def test_equal_scores_cannot_be_split(self):
        y=np.array([1,0,1,0]);s=np.full(4,.5)
        self.assertEqual(choose_threshold(y,s),.5)
        self.assertEqual(metrics(y,s>=.5)['recall'],1)

    def test_requires_positive_support(self):
        with self.assertRaises(ValueError): choose_threshold(np.array([0,0]),np.array([.1,.2]))


if __name__=='__main__': unittest.main()
