def group_by_info_qubit(result, num_info_qubits):
	groups = [{} for _ in range(2 ** num_info_qubits)]
	for bit_str, count in result.items():
		label = int(bit_str[-num_info_qubits:][::-1], 2)
		sub_bit_string = bit_str[:-num_info_qubits]
		groups[label][sub_bit_string] = count

	return groups
