import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class MyPlotter():
	def __init__(self, x_label="X Label", y_label="Y Label", title="No Title"):
		self.fig, self.ax = plt.subplots(1)
		self.ax.spines["top"].set_visible(False)
		self.ax.spines["bottom"].set_visible(False)
		self.ax.spines["right"].set_visible(False)
		self.ax.spines["left"].set_visible(False)
		self.ax.set_facecolor('#eaeaf2')
		plt.grid(color='#ffffff', linestyle='-', linewidth=1)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)

		plt.xlabel(x_label, fontsize=12)
		plt.ylabel(y_label, fontsize=12)
		plt.title(title, fontsize=16)

		self.data_arrays = []
		self.array_len = -1

		self.mean_array = []
		self.var_array = []
		self.max_array = []
		self.min_array = []

	def load_array(self, file_name_arrays, early_stop=None):
		data_arrays = [[np.genfromtxt(name, delimiter="\n", unpack=True) for name in array_set] for array_set in file_name_arrays]
		if(early_stop == None): self.array_len = min([min([len(el) for el in array_set]) for array_set in data_arrays])
		else: self.array_len = early_stop
		print(self.array_len)
		self.data_arrays = np.array([[el[:self.array_len] for el in array_set] for array_set in data_arrays], dtype=object)

	def render(self, labels, colors):
		err_msg = "load some data before the render!"
		assert self.array_len > 0, err_msg

		for mean_values, max_values, min_values, label, color in zip(self.mean_array, self.max_array, self.min_array, labels, colors):
			self.ax.plot(self.x_axes, mean_values, label=label, color=color, linestyle='-', linewidth=1.2 )
			self.ax.fill_between(self.x_axes, max_values, min_values, facecolor=color, alpha=0.3)

		self.ax.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=14)
		plt.show()

	def render_std(self, labels, colors):
		err_msg = "load some data before the render!"
		assert self.array_len > 0, err_msg

		for mean_values, var_values, label, color in zip(self.mean_array, self.var_array, labels, colors):
			self.ax.plot(self.x_axes, mean_values, label=label, color=color, linestyle='-', linewidth=1.2 )
			self.ax.fill_between(self.x_axes, mean_values+var_values, mean_values-var_values, facecolor=color, alpha=0.3)

		self.ax.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=14)
		plt.show()

	def process_data(self, rolling_window=1, starting_pointer=0, early_stop=None):
		rolling_queue = deque(maxlen=rolling_window)
		self.x_axes = [i for i in range(self.array_len-starting_pointer)]

		for array_set in self.data_arrays:
			for array in array_set:
				for i in range(self.array_len):
					rolling_queue.append(array[i])
					array[i] = np.mean(rolling_queue)
				rolling_queue.clear()

		# Fix for different array size
		self.data_arrays = np.array([np.array(el) for el in self.data_arrays], dtype=object)
		self.mean_array = np.array([[np.mean(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays])
		self.var_array = np.array([[np.std(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays])
		self.max_array = [[np.max(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays]
		self.min_array = [[np.min(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays]

