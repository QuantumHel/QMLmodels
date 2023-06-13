from qiskit import QuantumCircuit, QuantumRegister
import math

def combine_to_superposition(
	circuits: list[QuantumCircuit]
) -> QuantumCircuit:
	# TODO check that all circuits have the same num_qubits

	if len(circuits) == 1:
		return circuits[0]

	num_ancillas = math.log2(len(circuits))
	if not num_ancillas.is_integer():
			raise ValueError("The amount of circuits has to be a power of 2 for a superposition of circuits.")
	num_ancillas = int(num_ancillas)
	num_qubits = circuits[0].num_qubits + num_ancillas
	qr = QuantumRegister(num_qubits)
	circuit = QuantumCircuit(qr)

	controls = list(range(num_ancillas))
	targets = list(range(num_ancillas, num_qubits))
	circuit.h(controls)

	for index, circ in enumerate(circuits):
		bit_string_index = format(index, "b").zfill(num_ancillas)
		qubits_to_X = []
		for i, v in enumerate(bit_string_index):
			if v == "0":
				qubits_to_X.append(i)
		if len(qubits_to_X) > 0: circuit.x(qubits_to_X)
		gate = circ.to_gate().control(num_ancillas)
		circuit.append(gate, [*controls, *targets])
		if len(qubits_to_X) > 0: circuit.x(qubits_to_X)

	return circuit
