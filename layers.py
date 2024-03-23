import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from utils import Decorators, spike_generator, plot_heatmap
from neurons import IZHI
import os
import pandas as pd
from tqdm import tqdm
from utils import *


class InputLayer:
    def __init__(self):
        pass


class HiddenLayer:
    # ---parent class---
    def __init__(self):
        self.graph = nx.watts_strogatz_graph(10, 5, 0.5)
        self.node_opts = {"node_size": 500, "node_color": "r"}
        self.neurons = []
        self.dt = 0.01
        self.n = 10
        self.k = 5
        self.p = 0.5

        self.syn_matrix = np.zeros((self.n, self.n), dtype=np.float64)

        self.name = None

    def from_edges(self):
        pass

    def intra_forward(self, input_spike: np.ndarray):
        """
        Функция для расчета спайков для каждого шага
        :param input_spike:
        :return:
        """
        assert input_spike.shape[1] == len(self.syn_matrix), 'Размерности матриц должны совпадать!'
        output_spike = np.zeros([input_spike.shape[0], input_spike.shape[1]])

        membrane_potential = []

        for j in range(0, input_spike.shape[0]):
            for i in range(0, input_spike.shape[1]):
                self.neurons[i].step(self.dt, input_spike[j][i], 1)
                v = self.neurons[i].v

                # membrane_potential.append(v)

                if v >= self.neurons[i].thrs:
                    output_spike[j][i] = 10.0
                else:
                    continue

            input_sw = np.dot(a=output_spike, b=self.syn_matrix)

            for i in range(0, input_sw.shape[1]):
                self.neurons[i].step(self.dt, input_sw[j][i], 1)
                v = self.neurons[i].v

                membrane_potential.append(v)

                if v >= self.neurons[i].thrs:
                    output_spike[j][i] = 10.0
                else:
                    continue

        return output_spike, np.reshape(membrane_potential, (input_spike.shape[0], self.n))

    def draw_graph(self):
        # ---directed graph---
        g = nx.DiGraph(directed=True)

        edges_lst = []

        for i in range(0, self.n):
            for j in range(0, self.n):
                if self.syn_matrix[i][j] == 0:
                    continue
                else:
                    edge = (f'{i}', f'{j}')
                    edges_lst.append(edge)

        g.add_edges_from(edges_lst)
        pos = nx.circular_layout(g)

        nx.draw_networkx(g, pos=pos, with_labels=True, **self.node_opts)
        # nx.draw_networkx(self.graph, **self.node_opts)
        plt.savefig(f'{self.name}/graph.png', dpi=300, format='PNG')


class IntraConnectLayer(HiddenLayer):
    """

    """

    def __init__(self, amount_neurons: int, k_neighbours: int, probability: float, dtau: float, name: str) -> None:
        """
        Конструктор класса
        :param amount_neurons: (int) - общее число нейронов в слое
        :param k_neighbours: (int) - количество соседей для каждого нейрона (см. подробнее в алгоритме)
        :param probability: (float) - вероятность образования связей
        :param name: (str) - имя директории для сохранения результатов
        """
        super().__init__()
        assert amount_neurons >= k_neighbours, 'n must be > than k'
        assert probability >= 0 or probability < 1, 'probability must be in [0; 1)'

        self.n = amount_neurons
        self.k = k_neighbours
        self.p = probability

        self.dt = dtau

        self.name = name

        self.syn_matrix = np.zeros((self.n, self.n), dtype=np.float64)

        self.graph = nx.watts_strogatz_graph(self.n, self.k, self.p)

        self.neurons = []

        while len(self.neurons) < self.n:
            self.neurons.append(IZHI())

    @Decorators.benchmark
    def from_edges(self, double_percent=0.05) -> None:
        """

        :param pairs:
        :param double_percent:
        :return:
        """
        pairs = list(self.graph.edges)
        for pair in pairs:
            self.syn_matrix[pair[0], pair[1]] = 10.0
            # self.syn_matrix[pair[0], pair[1]] = np.round(np.random.rand(1)[0], 3)

        n_double = np.round(len(pairs) * double_percent)
        counter_double = 0

        pairs_copy = pairs.copy()

        # Добавление двунаправленных связей
        while counter_double < n_double:
            rand_edge_index = np.random.randint(low=0, high=len(pairs_copy), size=1)[0]
            choices_edge = pairs_copy[rand_edge_index]
            pairs_copy.pop(rand_edge_index)

            self.syn_matrix[choices_edge[1], choices_edge[0]] = np.round(np.random.rand(1)[0], 3)
            counter_double += 1

        print('Amount of double edges: ', np.round(len(pairs) * double_percent))

    def intra_forward(self, input_spike: np.ndarray):
        """
        Функция для расчета спайков для каждого шага
        :param input_spike:
        :return:
        """
        assert input_spike.shape[1] == len(self.syn_matrix), 'Размерности матриц должны совпадать!'
        output_spike = np.zeros([input_spike.shape[0], input_spike.shape[1]])

        membrane_potential = []
        spikes_indexes = [[] for _ in range(self.n)]

        for j in range(0, input_spike.shape[0]):
            for i in range(0, input_spike.shape[1]):
                self.neurons[i].step(self.dt, input_spike[j][i], 1)
                v = self.neurons[i].v

                # membrane_potential.append(v)

                if v >= self.neurons[i].thrs:
                    output_spike[j][i] = 10.0
                else:
                    continue

            input_sw = np.dot(a=output_spike, b=self.syn_matrix)

            for i in range(0, input_sw.shape[1]):
                self.neurons[i].step(self.dt, input_sw[j][i], 1)
                v = self.neurons[i].v

                membrane_potential.append(v)

                if v >= self.neurons[i].thrs:
                    output_spike[j][i] = 10.0
                    spikes_indexes[i].append(j)
                else:
                    continue

        # Функция dict_with_connections должна быть вне цикла!
        # При коррекции веса, если он был изменен с 10, она пропустит
        # эту пару нейронов

        connections = dict_with_connections(self.syn_matrix, 10)
        weight_corr_history = []

        output_spike = pd.DataFrame(output_spike)
        indexes = output_spike.apply(find_index_of_spikes, args=(10.0,))
        indexes.dropna(inplace=True)
        if indexes.empty == False:
            max_len = max(map(len, indexes))
            filled_lists = [list(filter(None, x)) + [np.nan] * (max_len - len(x)) for x in indexes]
            tau_max = find_tau_max(np.array(filled_lists), self.dt)
        else:
            tau_max = np.nan

        for key in connections.keys():
            if len(spikes_indexes[key]) == 1:
                post_spyke_step = spikes_indexes[key][-1]
                for pre_spyke_neuron in connections[key]:
                    for pre_spyke_step in spikes_indexes[pre_spyke_neuron]:
                        if pre_spyke_step > post_spyke_step:
                            # уменьшение связи по правилу STDP
                            continue
                        else:
                            coeff = intralayer_hebbian(post_spyke_step, pre_spyke_step, tau_max, 0.1, self.dt)
                            self.syn_matrix[pre_spyke_neuron][key] += coeff
                            weight_corr_history.append(f'Coef: {coeff}, Index: {pre_spyke_neuron, key}')

            elif len(spikes_indexes[key]) > 1:
                post_spyke_step = spikes_indexes[key][-1]
                limit = spikes_indexes[key][-2]
                for pre_spyke_neuron in connections[key]:
                    for pre_spyke_step in spikes_indexes[pre_spyke_neuron]:
                        if pre_spyke_step > limit:
                            if pre_spyke_step > post_spyke_step:
                                # уменьшение связи по правилу STDP
                                continue
                            else:
                                coeff = intralayer_hebbian(post_spyke_step, pre_spyke_step, 5, 0.1, self.dt)
                                self.syn_matrix[pre_spyke_neuron][key] += coeff
                                weight_corr_history.append(f'Coef: {coeff}, Index: {pre_spyke_neuron, key}')
                        else:
                            continue

            else:
                continue


        print(weight_corr_history)

        return (output_spike, np.reshape(membrane_potential, (input_spike.shape[0], self.n)),
                tau_max, pd.DataFrame(self.syn_matrix))


class IntraConnectLayerBio(HiddenLayer):
    """Биологически реалистичная топология малого мира *алгоритм в разработке*"""

    def __init__(self, amount_neurons: int, radius: float, probability_near: float, probability_far: float, name: str):
        """
        Конструктор класса
        :param amount_neurons: (int) - количество нейронов
        :param radius: (float) - радиус ближней зоны
        :param probability_near: (float) - вероятность образования связи между двумя нейронами в ближней области
        :param probability_far: (float) - вероятность образования связи между двумя нейронами в дальней зоне
        """
        super().__init__()
        self.n = amount_neurons
        self.r = radius
        self.p_near = probability_near
        self.p_far = probability_far

        self.syn_matrix = np.zeros((self.n, self.n), dtype=np.float64)

        self.name = name

        self.graph = nx.Graph()

        while len(self.neurons) < self.n:
            self.neurons.append(IZHI())

    @staticmethod
    def __generate_node_coord(n: int) -> pd.DataFrame:
        """
        Private method for generating node`s coordinates 2D

        :param n: (int) amount of neurons
        :return: None
        """

        df_coord = pd.DataFrame(columns=['x', 'y'])
        x_lst = []
        y_lst = []

        while len(x_lst) < n:
            x_rand = np.round(np.random.rand(1)[0] * 1000, 2)
            x_lst.append(x_rand)

            y_rand = np.round(np.random.rand(1)[0] * 1000, 2)
            y_lst.append(y_rand)

        df_coord['x'] = x_lst
        df_coord['y'] = y_lst

        # df_coord.to_csv(name_df, index=False)

        return df_coord

    @staticmethod
    def __calculate_dist(x_1: float, y_1: float, x_2: float, y_2: float) -> float:
        """
        Calculation of distance between 2 nodes in 2-dimensions surface

        :param x_1: (float)
        :param y_1: (float)
        :param x_2: (float)
        :param y_2: (float)
        :return:
        """

        d = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        return np.round(d, 2)

    @staticmethod
    def __synapse_connect(self, node_1, node_2, distance: float):
        # ---near area---
        if distance <= self.r and np.random.choice([True, False], 1, p=[self.p_near, 1 - self.p_near])[0]:
            syn_force = 5.0
            self.graph.add_edge(node_1, node_2, weight=syn_force, capacity=0)
            self.syn_matrix[int(node_1), int(node_2)] = syn_force

        # ---far area---
        elif distance > self.r and np.random.choice([True, False], 1, p=[self.p_far, 1 - self.p_far])[0]:
            syn_force = 1.0
            self.graph.add_edge(node_1, node_2, weight=syn_force, capacity=0)
            self.syn_matrix[int(node_1), int(node_2)] = syn_force

    @Decorators.benchmark
    def from_edges(self) -> None:
        df_coord = self.__generate_node_coord(self.n)
        df_coord.to_csv(f'{self.name}/neuron_coord_{self.name}.csv', index=False)

        neurons_index = np.arange(0, self.n)

        meshgrid = np.array(np.meshgrid(neurons_index, neurons_index)).T.reshape(-1, 2).tolist()
        pairs_lst = meshgrid.copy()

        for pair in meshgrid:
            if pair[0] == pair[1] or pair[1] < pair[0]:
                pairs_lst.remove(pair)
            else:
                continue

        for pair in pairs_lst:
            x_1 = df_coord['x'].loc[pair[0]]
            y_1 = df_coord['y'].loc[pair[0]]
            x_2 = df_coord['x'].loc[pair[1]]
            y_2 = df_coord['y'].loc[pair[1]]

            distance = self.__calculate_dist(x_1, y_1, x_2, y_2)
            print('Distance: ', distance)

            self.__synapse_connect(self,f'{pair[0]}', f'{pair[1]}', distance)

        print(self.graph)
        print('Neurons', nx.get_node_attributes(self.graph, 'pos'))
        print('Synapse: ', nx.get_edge_attributes(self.graph, 'weight'))

    @Decorators.benchmark
    def feedforward(self):
        pass


class FFLayer(HiddenLayer):
    def __init__(self):
        super().__init__()
        pass


# Рассписать распространение спайков в матричном виде
def main():
    n = 30
    layer = IntraConnectLayer(n, 10, 0.5)
    layer.from_edges()

    spike_series = spike_generator(n=n, steps=100, version='1')

    for i in tqdm(range(len(spike_series)), ascii=True, desc='forward'):
        out= layer.intra_forward(spike_series[i])
        # print(i)

    # layer.draw_graph()

    df_name = 'hidden_neurons/data_mem.csv'
    plot_heatmap(df_name)


if __name__ == '__main__':
    main()
