import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

"""Regression checks for invalid-output accounting; no GPU or dataset access."""
import unittest
from evaluate import summarize


class MetricTests(unittest.TestCase):
    def test_invalid_predictions_remain_in_denominator(self):
        rows = [dict(label=0, prediction=0, score=None, latency_ms=1),
                dict(label=1, prediction=-1, score=None, latency_ms=2)]
        result = summarize(rows)
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['toxic_recall'], 0)
        self.assertEqual(result['invalid_outputs'], 1)
        self.assertEqual(result['confusion_matrix'], [[1, 0, 0], [0, 0, 1]])
        self.assertIsNone(result['average_precision'])

    def test_perfect_labels_with_wrapper_are_separately_counted(self):
        rows = [dict(label=y, prediction=y, score=None, latency_ms=1,
                     raw_format_violation=True) for y in (0, 1)]
        result = summarize(rows)
        self.assertEqual(result['macro_f1'], 1)
        self.assertEqual(result['raw_format_violations'], 2)
        self.assertEqual(result['invalid_outputs'], 0)


if __name__ == '__main__':
    unittest.main()
