import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from utils import Decorators, spike_generator, plot_heatmap
from neurons import IZHI
import os
import pandas as pd
from tqdm import tqdm


class InputLayer:
    def __init__(self):
        pass


class HiddenLayer:
    # ---parent class---
    def __init__(self):
        self.graph = nx.watts_strogatz_graph(10, 5, 0.5)
        self.node_opts = {"node_size": 500, "node_color": "r"}
        self.n = 10
        self.k = 5
        self.p = 0.5

        self.syn_matrix = np.zeros((self.n, self.n), dtype=np.float64)

        self.name = None

    def from_edges(self):
        pass

    def feedforward(self):
        pass

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

    def __init__(self, amount_neurons: int, k_neighbours: int, probability: float, name: str) -> None:
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

    def intra_forward(self, input_spike: np.ndarray) -> np.ndarray:
        """
        Функция для расчета спайков для каждого шага
        :param input_spike:
        :return:
        """
        assert len(input_spike) == len(self.syn_matrix), 'Размерности матриц должны совпадать!'
        output_spike = np.zeros(input_spike.shape[0])

        dt = 0.01

        # ---ОПТИМИЗИРОВАТЬ---
        # Постоянное открытие датафрейма, возможно, сильно замедляет моделирование

        check_data = 'data_mem.csv' in os.listdir(f'{self.name}')  # Проверка датафрейма в директории
        if check_data:
            df = pd.read_csv(f'{self.name}/data_mem.csv')
        else:
            columns_spike = []
            columns_neuron = []
            columns_out_spike = []
            for i in range(len(input_spike)):
                columns_spike.append(f'inp_spike_{i}')
                columns_neuron.append(f'neuron_{i}')
                columns_out_spike.append(f'out_spike_{i}')

            df = pd.DataFrame(columns=columns_spike+columns_neuron+columns_out_spike)

        membrane_potential = []

        for i in range(0, len(input_spike)):
            self.neurons[i].step(dt, input_spike[i], 1)
            v = self.neurons[i].v

            # membrane_potential.append(v)

            if v >= self.neurons[i].thrs:
                output_spike[i] = 10.0
            else:
                continue

        input_sw = np.dot(a=output_spike, b=self.syn_matrix)

        for i in range(0, len(input_sw)):
            self.neurons[i].step(dt, input_sw[i], 1)
            v = self.neurons[i].v

            membrane_potential.append(v)

            if v >= self.neurons[i].thrs:
                output_spike[i] = 10.0
            else:
                continue

        df.loc[len(df.index)] = input_spike.tolist() + membrane_potential + output_spike.tolist()
        df.to_csv(f'{self.name}/data_mem.csv', index=False)

        return output_spike

    @Decorators.benchmark
    def feedforward(self, input_spike):
        """
        Функция для расчетов спайков
        :param input_spike: (np.array) - входные спайки
        :return:
        """
        output = []

        return output


class IntraConnectLayerBio(HiddenLayer):
    """Биологически реалистичная топология малого мира *алгоритм в разработке*"""

    def __init__(self, amount_neurons: int, radius: float, probability_near: float, probability_far: float):
        """
        Конструктор класса
        :param amount_neurons: (int)
        :param radius: (float)
        :param probability_near: (float)
        :param probability_far: (float)
        """
        super().__init__()
        self.n = amount_neurons
        self.r = radius
        self.p_near = probability_near
        self.p_far = probability_far

        self.syn_matrix = np.zeros((self.n, self.n), dtype=np.float64)

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

        :param x_1:
        :param y_1:
        :param x_2:
        :param y_2:
        :return:
        """

        d = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        return np.round(d, 2)

    @Decorators.benchmark
    def from_edges(self):
        df_coord = self.__generate_node_coord(self.n)



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

    spike_series = spike_generator(n=n, steps=1000, version='1')

    for i in tqdm(range(len(spike_series)), ascii=True, desc='forward'):
        out = layer.intra_forward(spike_series[i])
        # print(i)

    # layer.draw_graph()

    df_name = 'hidden_neurons/data_mem.csv'
    plot_heatmap(df_name)


if __name__ == '__main__':
    main()
