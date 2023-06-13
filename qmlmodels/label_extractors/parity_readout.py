def parity_readout(counts):
	"""
	Gives:
		0 when an even number of 1:s
		1 when an uneven number of 1:s
	"""	
	total_count = 0
	even_count = 0
	for bit_string, count in counts.items():
		total_count += count
		if (bit_string.count("1") % 2) == 0:
			even_count += count
	even = even_count / total_count
	return [even, 1 - even]
