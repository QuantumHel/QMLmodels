from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile
from scipy.optimize import minimize
from qmlmodels.utils import combine_to_superposition, group_by_label, group_by_info_qubit, combine_circuit, per_shot_label_extractor
import random
import math
import numpy as np

class VCC:
	def __init__(
		self,
		feature_map: QuantumCircuit,
		ansatz: QuantumCircuit,
		label_extractor,
		loss_function,
		optimize_method: str = "COBYLA",
		shots = 1024,
		initial_parameters = None
	):
		"""
		LIMIT: only number labels starting from 0 allowed.
			could do that map to these and back
		TODO
		"""
		self._feature_map = feature_map
		self._ansatz = ansatz
		self._label_extractor = label_extractor
		self._loss_function = loss_function
		self._optimize_method = optimize_method
		self._shots = shots
		self._parameters = None
		if initial_parameters is not None:
			self._parameters = initial_parameters
		self._simulator = Aer.get_backend('aer_simulator')
		# simulator could be parameter and options for multithread and gpu
		self._trained = False

		self._num_qubits = max(feature_map.num_qubits, ansatz.num_qubits)
		qr = QuantumRegister(self._num_qubits)
		cr = ClassicalRegister(self._num_qubits)
		self._circuit = QuantumCircuit(qr, cr)
		self._circuit.compose(self._feature_map, inplace=True)
		self._circuit.compose(self._ansatz, inplace=True)
		self._circuit.measure(list(range(self._num_qubits)), list(range(self._num_qubits)))
		self._transpiled_circuit = transpile(self._circuit, self._simulator)

		# This is for counting iterations for scipy that sometimes does not do it.
		self._training_iteration_counter = 0

	def train(self, X, y, shots = None, training_mode = "normal"):
		"""
		TODO
		training modes:
			normal, label_superpositions, full_superposition
		"""
		if self._parameters is None:
			self._parameters = [random.random() for _ in range(self._ansatz.num_parameters)]
		
		match training_mode:
			case "normal":
				loss = self._normal_loss(X, y, shots)
			case "label_superpositions":
				loss = self._label_superpositions_loss(X, y, shots)
			case "label_superpositions_per_shot_loss":
				loss = self._label_superpositions_loss(X, y, shots, True)
			case "classical_label_superpositions":
				loss = self._classical_label_superpositions_loss(X, y, shots)
			case "full_superposition":
				loss = self._full_superposition_loss(X, y, shots)
			case "classical_full_superposition":
				loss = self._classical_full_superpositions_loss(X, y, shots)
			case _:
				raise ValueError("training_mode has to be 'training_mode', 'label_superpositions', or 'full_superposition'.")

		self._training_iteration_counter = 0
		def callback(x, *args):
			self._training_iteration_counter += 1

		result = minimize(
			loss,
			self._parameters,
			method=self._optimize_method,
			callback=callback
		)

		if not hasattr(result, 'nit'):
			result.nit = self._training_iteration_counter

		self._parameters = result.x
		self._trained = True
		return result

	def predict(self, X, shots = None):
		"""
		TODO
		"""
		self._check_if_trained()
		return self._predict(X, shots=shots)

	def accuracy(self, X, y, shots = None):
		"""
		TODO
		"""
		predictions = self.predict(X)
		correct = 0
		for label, prediction in zip(y, predictions):
			predicted_label = prediction.index(max(prediction))
			if predicted_label == label:
				correct += 1
		return correct / len(y)

	def _normal_loss(self, X, y, shots):
		def loss(parameters):
			return self._loss_function(y, self._predict(X, parameters, shots))
		return loss

	def _label_superpositions_loss(self, X, y, shots, per_shot_loss=False):
		groups = group_by_label(X, y)
		circuits = []
		for label, data in groups.items():
			feature_circuits = [self._feature_map.bind_parameters(d) for d in data]
			circuit = combine_to_superposition(feature_circuits)
			circuit = combine_circuit(circuit, self._ansatz)
			circuit = transpile(circuit, self._simulator)
			circuits.append([label, circuit])

		def loss(parameters):
			labels = []
			histograms = []
			for label, circuit in circuits:
				labels.append(label)
				circ = circuit.bind_parameters(parameters)
				job = self._simulator.run(circ, shots=shots)
				result = job.result().get_counts()
				histograms.append(result)

			if per_shot_loss:
				spread_labels, predicted = per_shot_label_extractor(
					labels,
					histograms,
					self._label_extractor
				)
				return self._loss_function(spread_labels, predicted)
			else:
				predicted = [self._label_extractor(hist) for hist in histograms]
				return self._loss_function(labels, predicted)

		return loss

	def _classical_label_superpositions_loss(self, X, y, shots):
		groups = group_by_label(X, y)

		def sample_counts(l, count):
			selected_counts = [0 for _ in list(range(l))]
			choices = np.random.choice(a=list(range(l)), size=shots, replace=True)
			for choice in choices:
				selected_counts[choice] += 1
			return selected_counts

		def combine_hist(results):
			qubits = len(list(results[0].keys())[0])
			result = {}
			for i in range(2 ** qubits):
				bit_string = format(i, "b").zfill(qubits)
				result[bit_string] = 0
			for res in results:
				for bit_string, count in res.items():
					result[bit_string] += count
			return result

		def loss(parameters):
			labels = []
			predicted = []
			for label, group in groups.items():
				labels.append(label)
				selected = sample_counts(len(group), shots)

				results = []
				for x, count in zip(group, selected):
					if count < 0: continue
					circuit = combine_circuit(
						self._feature_map.bind_parameters(x),
						self._ansatz.bind_parameters(parameters)
					)
					circuit = transpile(circuit, self._simulator)
					job = self._simulator.run(circuit, shots=count)
					results.append(job.result().get_counts())
				predicted.append(self._label_extractor(combine_hist(results)))

			return self._loss_function(labels, predicted)

		return loss

	def _full_superposition_loss(self, X, y, shots):
		groups = group_by_label(X, y)
		labels = []
		circuits = []
		for label, data in groups.items():
			feature_circuits = [self._feature_map.bind_parameters(d) for d in data]
			circuit = combine_to_superposition(feature_circuits)
			labels.append(label)
			circuits.append(circuit)

		info_qubits = int(math.log2(len(circuits)))
		circuit = combine_to_superposition(circuits)
		circuit = combine_circuit(circuit, self._ansatz, info_qubits)
		circuit = transpile(circuit, self._simulator)

		def loss(parameters):
			circ = circuit.bind_parameters(parameters)
			job = self._simulator.run(circ, shots=shots)
			result = job.result().get_counts()
			histograms = group_by_info_qubit(result, info_qubits)
			predicted = [self._label_extractor(hist) for hist in histograms]
			return self._loss_function(labels, predicted)

		return loss

	def _classical_full_superpositions_loss(self, X, y, shots):
		raise NotImplementedError("Classical full superpositions loss training is not yet implemented.")

	def _check_if_trained(self):
		if not self._trained:
			print("Warning: the models is not trained, trying to use initial parameters.")
		if self._parameters is None:
			raise AttributeError("To run a model, it needs to be trained or initial parameters need to be given.")

	def _predict(self, X, parameters = None, shots = None):
		if shots is None: shots = self._shots
		if parameters is None:
			parameters = self._parameters
		parameter_binds = self._create_parameter_binds(X, parameters)

		job = self._simulator.run([self._transpiled_circuit] * len(X), shots=shots, parameter_binds=parameter_binds)
		results = job.result().get_counts()
		predicted = [self._label_extractor(result) for result in results]
		return predicted

	def _create_parameter_binds(self, X, parameters):
		# TODO see if can be moved to utils
		num_samples = len(X)
		ansatz_binds = {}
		for index, parameter in enumerate(self._ansatz.parameters):
			ansatz_binds[parameter] = [parameters[index]]
		if len(X) == 0:
			return ansatz_binds

		parameter_binds = [ansatz_binds.copy() for _ in range(num_samples)]
		for index, parameter in enumerate(self._feature_map.parameters):
			for i, parameter_bind in enumerate(parameter_binds):
				parameter_bind[parameter] = [X[i][index]]

		return parameter_binds
