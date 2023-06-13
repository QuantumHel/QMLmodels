def group_by_label(elements: list, labels: list[int]):
	groups = {}
	for label in labels:
		groups[label] = []

	for element, label in zip(elements, labels):
		groups[label].append(element)

	return groups
