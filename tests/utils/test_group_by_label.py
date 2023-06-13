import unittest
import pytest
from qmlmodels.utils import group_by_label

class TestGroupByLabel(unittest.TestCase):
	def test_basic1(self):
		elements = [1, 2, 3, 4, 5, 6]
		labels = [1, 2, 3, 1, 2, 3]
		goal = {1: [1, 4], 2: [2, 5], 3: [3, 6]}
		answer = group_by_label(elements, labels)
		self.assertEqual(goal, answer)

	def test_basic2(self):
		elements = [1, 2, 3, 4, 5, 6]
		labels = [1, 2, 2, 1, 2, 2]
		goal = {1: [1, 4], 2: [2, 3, 5, 6]}
		answer = group_by_label(elements, labels)
		self.assertEqual(goal, answer)
