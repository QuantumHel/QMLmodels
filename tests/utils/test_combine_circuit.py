import unittest
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, ClassicalRegister
from qmlmodels.utils import combine_circuit

class TestCombineCircuit(unittest.TestCase):
	# These tests do not use parameters, as qiskit does not
	# have comparison of circuits implemented ://///
	# Gasm convertion does not support paramaters
	# Also bits are added to goal after making the circuit,
	# because the qasm conversion for some reason treats this
	# differently.

	def test_same_size_feature_map(self):
		feature_map = QuantumCircuit(2)
		feature_map.rz(0.5, 0)
		feature_map.rz(0.2, 1)

		a = 0.3
		ansatz = QuantumCircuit(2)
		ansatz.rx(a, 0)
		ansatz.cx(0, 1)

		goal = QuantumCircuit(2)
		goal.add_bits(ClassicalRegister(2))
		goal.rz(0.5, 0)
		goal.rz(0.2, 1)
		goal.rx(a, 0)
		goal.cx(0, 1)
		goal.measure([0, 1], [0, 1])

		self.assertEqual(goal.qasm(), combine_circuit(feature_map, ansatz).qasm())


	def test_more_qubits_feature_map(self):
		feature_map = QuantumCircuit(4)
		feature_map.rz(0.5, 0)
		feature_map.rz(0.2, 1)
		feature_map.rz(0.4, 2)
		feature_map.rz(0.3, 3)
		feature_map.cx(2, 0)
		feature_map.cx(3, 1)

		a = 0.3
		ansatz = QuantumCircuit(2)
		ansatz.rx(a, 0)
		ansatz.cx(0, 1)

		goal = QuantumCircuit(4)
		goal.add_bits(ClassicalRegister(2))
		goal.rz(0.5, 0)
		goal.rz(0.2, 1)
		goal.rz(0.4, 2)
		goal.rz(0.3, 3)
		goal.cx(2, 0)
		goal.cx(3, 1)

		goal.rx(a, 2)
		goal.cx(2, 3)
		goal.measure([2, 3], [0, 1])

		self.assertEqual(goal.qasm(), combine_circuit(feature_map, ansatz).qasm())

	def test_all_extra_as_info_qubits(self):
		feature_map = QuantumCircuit(4)
		feature_map.rz(0.5, 0)
		feature_map.rz(0.2, 1)
		feature_map.rz(0.4, 2)
		feature_map.rz(0.3, 3)
		feature_map.cx(2, 0)
		feature_map.cx(3, 1)

		a = 0.3
		ansatz = QuantumCircuit(2)
		ansatz.rx(a, 0)
		ansatz.cx(0, 1)

		goal = QuantumCircuit(4)
		goal.add_bits(ClassicalRegister(4))
		goal.rz(0.5, 0)
		goal.rz(0.2, 1)
		goal.rz(0.4, 2)
		goal.rz(0.3, 3)
		goal.cx(2, 0)
		goal.cx(3, 1)

		goal.rx(a, 2)
		goal.cx(2, 3)
		goal.measure([0, 1, 2, 3], [0, 1, 2, 3])

		result = combine_circuit(feature_map, ansatz, 2).qasm()
		self.assertEqual(goal.qasm(), result)

	def test_some_as_extra_qubits(self):
		feature_map = QuantumCircuit(4)
		feature_map.rz(0.5, 0)
		feature_map.rz(0.2, 1)
		feature_map.rz(0.4, 2)
		feature_map.rz(0.3, 3)
		feature_map.cx(2, 0)
		feature_map.cx(3, 1)

		a = 0.3
		ansatz = QuantumCircuit(2)
		ansatz.rx(a, 0)
		ansatz.cx(0, 1)

		goal = QuantumCircuit(4)
		goal.add_bits(ClassicalRegister(3))
		goal.rz(0.5, 0)
		goal.rz(0.2, 1)
		goal.rz(0.4, 2)
		goal.rz(0.3, 3)
		goal.cx(2, 0)
		goal.cx(3, 1)

		goal.rx(a, 2)
		goal.cx(2, 3)
		goal.measure([0, 2, 3], [0, 1, 2])

		result = combine_circuit(feature_map, ansatz, 1).qasm()
		self.assertEqual(goal.qasm(), result)
