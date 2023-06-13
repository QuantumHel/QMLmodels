import unittest
import pytest
from qmlmodels.label_extractors import parity_readout

class TestParityReadout(unittest.TestCase):
	def test_basic1(self):
		counts = {
			"000": 10, # even
			"001": 10,
			"010": 10,
			"011": 6,  # even
			"100": 10,
			"101": 12, # even
			"110": 32, # even
			"111": 10,
		}
		goal = [0.6, 0.4]
		self.assertEqual(goal, parity_readout(counts))

	def test_basic2(self):
		counts = {
			"000": 7,  # even
			"001": 20,
			"010": 24,
			"011": 4,  # even
			"100": 16,
			"101": 3,  # even
			"110": 5,  # even
			"111": 21,
		}
		goal = [0.19, 0.81]
		self.assertEqual(goal, parity_readout(counts))
