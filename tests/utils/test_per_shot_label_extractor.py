import unittest
import pytest
from qmlmodels.utils import per_shot_label_extractor

class TestPerShotLabelExtractor(unittest.TestCase):
	def test_basic(self):
		labels = [0, 1, 2]
		histograms = [{
			"00": 1, "01": 0, "10": 1, "11": 1,
		}, {
			"00": 2, "01": 2, "10": 0, "11": 0,
		}, {
			"00": 0, "01": 1, "10": 2, "11": 1,
		}]
		def label_extractor(hist):
			values = 0
			summ = 0
			for bit_str, count in hist.items():
				summ += count * int(bit_str,  2)
				values += count
			
			return summ / values

		goal_labels = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
		goal_predicted = [0, 2, 3, 0, 0, 1, 1, 1, 2, 2, 3]
		got_labels, got_predicted = per_shot_label_extractor(
			labels,
			histograms,
			label_extractor
		)

		self.assertEqual(goal_labels, got_labels)
		self.assertEqual(goal_predicted, got_predicted)
