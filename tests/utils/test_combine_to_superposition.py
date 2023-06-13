import unittest
import pytest
from qmlmodels.utils import combine_to_superposition
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

class TestCombineToSuperposition(unittest.TestCase):
	def test_basic(self):
		num_qubits = 2
		circ1 = QuantumCircuit(QuantumRegister(num_qubits))
		circ1.x(0)
		circ1.cnot(0, 1)

		circ2 = QuantumCircuit(QuantumRegister(num_qubits))
		circ2.h(0)
		circ2.cnot(0, 1)

		circ3 = QuantumCircuit(QuantumRegister(num_qubits))
		circ3.x(0)
		circ3.cnot(1, 0)

		circ4 = QuantumCircuit(QuantumRegister(num_qubits))
		circ4.h(0)
		circ4.cnot(1, 0)

		circuits = [circ1, circ2, circ3, circ4]

		goal = QuantumCircuit(QuantumRegister(num_qubits + 2))
		goal.h([0, 1])
		goal.x([0, 1])
		goal.append(circ1.to_gate().control(2), [0, 1, 2, 3])
		goal.x([0, 1])

		goal.x([0])
		goal.append(circ2.to_gate().control(2), [0, 1, 2, 3])
		goal.x([0])

		goal.x([1])
		goal.append(circ3.to_gate().control(2), [0, 1, 2, 3])
		goal.x([1])

		goal.append(circ4.to_gate().control(2), [0, 1, 2, 3])

		result = combine_to_superposition(circuits)
		self.assertTrue(Statevector.from_instruction(goal).equiv(Statevector.from_instruction(result)))

	def test_circuit_count_error(self):
		num_qubits = 2
		circ1 = QuantumCircuit(QuantumRegister(num_qubits))
		circ1.x(0)
		circ1.cnot(0, 1)

		circ2 = QuantumCircuit(QuantumRegister(num_qubits))
		circ2.h(0)
		circ2.cnot(0, 1)

		circ3 = QuantumCircuit(QuantumRegister(num_qubits))
		circ3.x(0)
		circ3.cnot(1, 0)

		circuits = [circ1, circ2, circ3]

		self.assertRaises(ValueError, combine_to_superposition, circuits)

	def test_single_circuit_returns_itself(self):
		num_qubits = 2
		circ1 = QuantumCircuit(QuantumRegister(num_qubits))
		circ1.x(0)
		circ1.cnot(0, 1)

		result = combine_to_superposition([circ1])
		self.assertEqual(circ1, result)
