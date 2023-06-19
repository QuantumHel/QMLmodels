from qiskit import ClassicalRegister

def combine_circuit(feature_map, ansatz, num_info_qubits=0):
	info_qubits = list(range(num_info_qubits))
	targets = list(range(feature_map.num_qubits))[-ansatz.num_qubits:]
	c_bits = list(range(num_info_qubits + ansatz.num_qubits))

	circ = feature_map.compose(ansatz, qubits=targets)
	cr = ClassicalRegister(num_info_qubits + ansatz.num_qubits)
	circ.add_bits(cr)
	circ.measure([*info_qubits, *targets], c_bits)
	return circ

if __name__ == "__main__":
	from qiskit.circuit import Parameter
	from qiskit import QuantumCircuit

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

	goal.rx(a, 0)
	goal.cx(0, 1)
	goal.measure([2, 3], [0, 1])

	print(goal.qasm())
	print(combine_circuit(feature_map, ansatz).qasm())
