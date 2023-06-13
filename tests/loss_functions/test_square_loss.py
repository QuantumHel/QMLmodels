import unittest
import pytest
from qmlmodels.loss_functions import square_loss

class TestSquareLoss(unittest.TestCase):
	def test_basic1(self):
		labels = [0, 1, 2, 1, 0]
		predictions = [
			[0.50, 0.25, 0.25],
			[0.25, 0.50, 0.25],
			[0.25, 0.25, 0.50],
			[0.25, 0.50, 0.25],
			[0.50, 0.25, 0.25],
		]
		self.assertEqual(0.25, square_loss(labels, predictions))

	def test_basic2(self):
		labels = [0, 0, 1]
		predictions = [
			[0.70, 0.30],
			[0.40, 0.60],
			[0.30, 0.70],
		]
		assert 0.18 == pytest.approx(square_loss(labels, predictions))

	def test_correct(self):
		labels = [0, 1, 2, 1, 0]
		predictions = [
			[1.00, 0.00, 0.00],
			[0.00, 1.00, 0.00],
			[0.00, 0.00, 1.00],
			[0.00, 1.00, 0.00],
			[1.00, 0.00, 0.00],
		]
		self.assertEqual(0, square_loss(labels, predictions))

	def test_wrong(self):
		labels = [0, 1, 2, 1, 0]
		predictions = [
			[0.00, 0.50, 0.50],
			[0.50, 0.00, 0.50],
			[0.50, 0.50, 0.00],
			[0.50, 0.00, 0.50],
			[0.00, 0.50, 0.50],
		]
		self.assertEqual(1, square_loss(labels, predictions))
