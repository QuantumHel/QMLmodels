def square_loss(labels, predictions):
	loss = 0
	for label, prediction in zip(labels, predictions):
		loss += (1 - prediction[label]) ** 2
	return loss / len(labels)
