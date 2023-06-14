import unittest
from qmlmodels.utils import group_by_info_qubit

class TestGroupByInfoQubit(unittest.TestCase):
	def test_basic(self):
		inputs = {
			# q3 q2 q1 q0
			# label 0
			"0000": 0,
			"0100": 1,
			"1000": 2,
			"1100": 3,
			# label 1
			"0010": 4,
			"0110": 5,
			"1010": 6,
			"1110": 7,
			# label 2
			"0001": 8,
			"0101": 9,
			"1001": 10,
			"1101": 11,
			# label 3
			"0011": 12,
			"0111": 13,
			"1011": 14,
			"1111": 15,
		}

		goal = [
			{"00": 0, "01": 1, "10": 2, "11": 3},
			{"00": 4, "01": 5, "10": 6, "11": 7},
			{"00": 8, "01": 9, "10": 10, "11": 11},
			{"00": 12, "01": 13, "10": 14, "11": 15},
		]

		self.assertEqual(goal, group_by_info_qubit(inputs, 2))
