# %% Preparing parameters
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qmlmodels.label_extractors import parity_readout
from qmlmodels.loss_functions import square_loss
from qmlmodels import VCC
import numpy as np

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
runs = 4


def run_experiment(training_mode, training_shots):
	train_accuracy = np.zeros(runs)
	test_accuracy = np.zeros(runs)
	iterations = np.zeros(runs)
	for i in range(runs):
		vcc = VCC(feature_map, ansatz, parity_readout, square_loss, shots=shots)
		res = vcc.train(train_features, train_labels, shots=training_shots, training_mode=training_mode)
		train_accuracy[i] = vcc.accuracy(train_features, train_labels)
		test_accuracy[i] = vcc.accuracy(test_features, test_labels)
		iterations[i] = res.nit
		print(".", end="")
	print()
	print(f"Training using '{training_mode}' and {training_shots} shots")
	print("Train accuracy mean:", np.mean(train_accuracy), ". With std of:", np.std(train_accuracy))
	print("Test accuracy mean:", np.mean(test_accuracy), ". With std of:", np.std(test_accuracy))
	print()

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
run_experiment("normal", shots)

# %%
# Normal training with shot count 1
run_experiment("normal", 1)

# %%
# Label superposition training
run_experiment("label_superpositions", shots)

# %%
# Label superposition training, but superposition is calculated with classical help
# TODO

# %%
# Full superposition training
run_experiment("full_superposition", shots)

# %%
# Full superposition training, but superposition is calculated with classical help
# TODO
