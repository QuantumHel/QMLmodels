def per_shot_label_extractor(labels, histograms, label_extractor):
	spread_labels = []
	predicted = []

	for label, histogram in zip(labels, histograms):
		for bit_str, count in histogram.items():
			pred = label_extractor({bit_str: 1})
			for _ in range(count):
				spread_labels.append(label)
				predicted.append(pred)

	return spread_labels, predicted
