import matplotlib.pyplot as plt
import numpy as np
import time
import itertools


class STDP:

    def __init__(self, A_plus, A_minus, tau_plus, tau_minus, tau_bound) -> None:
        self.A_plus = 5
        self.A_minus = 3
        self.tau_plus = 1.3
        self.tau_minus = 0.75
        self.tau_bound = 2

    def prepare_for_stdp(self, input_spike_time: np.ndarray) -> list:
        """Превращение дискретного ряда во времена спайков

        Args:
            input_spike_time (np.ndarray): Одномерный массив со спайками

        Returns:
            List: Время, когда нейрон спайкует
        """
        input_spikes_time = []
        for i in input_spike_time:
            if input_spike_time[i] == 10:
                input_spikes_time.append(i)
        return input_spikes_time

    def get_dw(self, spike_time_pre, spike_time_post, dt) -> float:
        """Корректировка веса

        Args:
            spike_time_pre (_type_): Время спайкования пресинаптического нейрона
            spike_time_post (_type_): Время спайкования постсинаптического нейрона
            dt (_type_): _description_

        Returns:
            float:
        """
        dw = 0
        delta = (spike_time_pre - spike_time_post) * dt

        if abs(delta) < self.tau_bound:
            if delta < 0:
                dw = -self.A_minus * np.exp(delta / self.tau_minus)
            elif delta > 0:
                dw = self.A_plus * np.exp(-delta / self.tau_plus)
            else:
                return dw

        return dw

    def generate_pairs(self, list_1, list_2) -> list:
        """Генерация всех возможных пар спайков из двух масивов

        Args:
            list_1 (_type_): Массив с временами спайкования пресинаптического нейрона
            list_2 (_type_): Массив с временами спайкования постсинаптического нейрона

        Returns:
            list:
        """
        pairs = []
        for r in itertools.product(list_1, list_2):
            pairs.append([r[0], r[1]])
        return pairs

    def normalize(self, previous_weights, current_weights):
        """Генерация всех возможных пар спайков из двух масивов

        Args:
            previous_weights (_type_): Веса после предыдущей итерации обучения
            current_weights (_type_): Веса после текущей итерации обучения

        """
        divisor = sum(current_weights)/sum(previous_weights)
        for i in range(len(current_weights)):
            current_weights[i]/divisor

    def stdp_plot(self, dw):
        x_init = np.linspace(0, 10, 10)
        plt.scatter(x_init, dw, color='red')
        plt.ylabel('dw')
        plt.xlabel('neuron')
        plt.show()


optimizer = STDP()

spike_matrix = np.array()  # матрица с временами всех спайков, spike_matrix[0] - pre, spike_matrix[1] - post

syn_matrix = np.array() # матрица с весами нейронов в слое

dw_matrix = np.array()  # матрица для обновления весов

spike_pairs = optimizer.generate_pairs(spike_matrix[0], spike_matrix[1])  

num_epochs = 1

dt = 0.1


def main():
    for epoch in range(num_epochs):
        previous_weights = syn_matrix.copy()
        for i in range(len(spike_pairs)):
            dw = optimizer.get_dw(spike_pairs[i][0], spike_pairs[i][1], dt)
            dw_matrix[i] = dw
        syn_matrix + dw_matrix
        optimizer.normalize(previous_weights, syn_matrix)

