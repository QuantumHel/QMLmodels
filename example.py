# %% Preparing parameters
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qmlmodels.label_extractors import parity_readout
from qmlmodels.loss_functions import square_loss
from qmlmodels import VCC

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

iris_data = load_iris()
features = iris_data.data[:100,:]
labels = iris_data.target[:100]
features = MinMaxScaler().fit_transform(features)

train_features, test_features, train_labels, test_labels = train_test_split(
	features, labels, train_size=64, random_state=123, stratify=labels
)

num_features = features.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_features)
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
shots = 1024
runs = 10

# %%
# Classical SVM
from sklearn.svm import SVC

svc = SVC()
svc.fit(train_features, train_labels)
svc_train_c4 = svc.score(train_features, train_labels)
svc_test_c4 = svc.score(test_features, test_labels)

print(f"Classical SVC on the training dataset: {svc_train_c4:.2f}")
print(f"Classical SVC on the test dataset:     {svc_test_c4:.2f}")

# %%
# Normal training
import numpy as np
train_accuracy = np.zeros(runs)
test_accuracy = np.zeros(runs)
for i in range(runs):
	vcc = VCC(feature_map, ansatz, parity_readout, square_loss, shots=shots)
	vcc.train(train_features, train_labels, shots=shots, training_mode="normal")
	train_accuracy[i] = vcc.accuracy(train_features, train_labels)
	test_accuracy[i] = vcc.accuracy(test_features, test_labels)
	print(".", end="")
print()

print("Training with normal VCC with", shots, "shots")
print("Train accuracy mean:", np.mean(train_accuracy), ". With std of:", np.std(train_accuracy))
print("Test accuracy mean:", np.mean(test_accuracy), ". With std of:", np.std(test_accuracy))
print()

# %%
# Normal training with shot count 1
train_accuracy = np.zeros(runs)
test_accuracy = np.zeros(runs)
for i in range(runs):
	vcc = VCC(feature_map, ansatz, parity_readout, square_loss, shots=shots)
	vcc.train(train_features, train_labels, shots=1, training_mode="normal")
	train_accuracy[i] = vcc.accuracy(train_features, train_labels)
	test_accuracy[i] = vcc.accuracy(test_features, test_labels)
	print(".", end="")
print()

print("Training with normal VCC with", 1, "shots")
print("Train accuracy mean:", np.mean(train_accuracy), ". With std of:", np.std(train_accuracy))
print("Test accuracy mean:", np.mean(test_accuracy), ". With std of:", np.std(test_accuracy))
print()
# %%
# Label superposition training
train_accuracy = np.zeros(runs)
test_accuracy = np.zeros(runs)
for i in range(runs):
	vcc = VCC(feature_map, ansatz, parity_readout, square_loss, shots=shots)
	vcc.train(train_features, train_labels, shots=shots, training_mode="label_superpositions")
	train_accuracy[i] = vcc.accuracy(train_features, train_labels)
	test_accuracy[i] = vcc.accuracy(test_features, test_labels)
	print(".", end="")
print()

print("Training with label specific superpositions with", shots, "shots")
print("Train accuracy mean:", np.mean(train_accuracy), ". With std of:", np.std(train_accuracy))
print("Test accuracy mean:", np.mean(test_accuracy), ". With std of:", np.std(test_accuracy))
print()

# %%
# Label superposition training, but superposition is calculated with classical help
# TODO

# %%
# Full superposition training
vcc = VCC(feature_map, ansatz, parity_readout, square_loss, shots=shots)
vcc.train(train_features, train_labels, shots=shots, training_mode="full_superposition")
print(vcc.accuracy(train_features, train_labels))
print(vcc.accuracy(test_features, test_labels))

# %%
# Full superposition training, but superposition is calculated with classical help
# TODO
