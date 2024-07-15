from brian2 import *
import matplotlib.pyplot as plt
from collections import defaultdict


class Visualization_SNN():

    @staticmethod
    def plot_w(S1M, gmax):

        '''
        Этот метод строит графики весов синапсов и потенциалов до и после спайков.

        Параметры:

        S1M: Монитор состояния синапсов.
        gmax: Максимальная проводимость синапсов.
        '''


        plt.rcParams["figure.figsize"] = (20, 10)
        subplot(311)
        plot(S1M.t / ms, S1M.w.T / gmax)
        ylabel('w / wmax')
        subplot(312)
        plot(S1M.t / ms, S1M.Apre.T)
        ylabel('apre')
        subplot(313)
        plot(S1M.t / ms, S1M.Apost.T)
        ylabel('apost')
        tight_layout()
        show();

    @staticmethod
    def plot_v(ESM, ISM, v_thresh_e, v_thresh_i, neuron=13):

        '''
        Этот метод строит графики мембранных потенциалов для возбудительных и тормозных нейронов.

        Параметры:

        ESM: Монитор состояния возбудительных нейронов.
        ISM: Монитор состояния тормозных нейронов.
        v_thresh_e: Пороговый потенциал для возбудительных нейронов.
        v_thresh_i: Пороговый потенциал для тормозных нейронов.
        neuron(Optional(int), default:13): Индекс нейрона для построения графика .
        '''

        plt.rcParams["figure.figsize"] = (20, 6)
        cnt = -50000  # tail
        plot(ESM.t[cnt:] / ms, ESM.v[neuron][cnt:] / mV, label='exc', color='r')
        plot(ISM.t[cnt:] / ms, ISM.v[neuron][cnt:] / mV, label='inh', color='b')
        plt.axhline(y=v_thresh_e / mV, color='pink', label='v_thresh_e')
        plt.axhline(y=v_thresh_i / mV, color='silver', label='v_thresh_i')
        legend()
        ylabel('v')
        show();

    @staticmethod
    def plot_rates(ERM, IRM):

        '''
        Этот метод строит графики частот спайков для возбудительных и тормозных нейронов.

        Параметры:

        ERM: Монитор частоты спайков возбудительных нейронов.
        IRM: Монитор частоты спайков тормозных нейронов.
        '''

        plt.rcParams["figure.figsize"] = (20, 6)
        plot(ERM.t / ms, ERM.smooth_rate(window='flat', width=0.1 * ms) * Hz, color='r')
        plot(IRM.t / ms, IRM.smooth_rate(window='flat', width=0.1 * ms) * Hz, color='b')
        ylabel('Rate')
        show();

    @staticmethod
    def plot_all_spikes(ESP, path, time):

        '''
        Этот метод строит график всех спайков возбудительных нейронов за первые 100 мс симуляции.

        Параметры:

        ESP: Монитор спайков возбудительных нейронов.
        path: Путь для сохранения графика.
        time: Время для визуализации.
        '''

        plt.rcParams["figure.figsize"] = (20, 6)
        scatter([spike_time for spike_time in ESP.t / ms if spike_time <= ESP.t[0] / ms + time],
                [neuron for neuron, time in zip(ESP.i, ESP.t / ms) if time <= ESP.t[0] / ms + time], label='exc')
        ylabel('Neuron index')
        legend()
        savefig(path)
        close();

    @staticmethod
    def plot_spikes(ESP, ISP, path, color_neurons=None):

        '''
        Этот метод строит график спайков возбудительных и тормозных нейронов,
        выделяя нейроны из самой большой группы.

        Параметры:

        ESP: Монитор спайков возбудительных нейронов.
        ISP: Монитор спайков тормозных нейронов.
        path: Путь для сохранения графика.
        color_neurons: Список нейронов для выделения цветом.
        '''

        plt.rcParams["figure.figsize"] = (20, 6)
        # vplot(ESP.t / ms, ESP.i, '.b', label='exc')
        if ISP is not None:
            plot(ISP.t / ms, ISP.i, '.g', label='inh')
        positions = [index for index, value in enumerate(ESP.i) if value in color_neurons]
        spike_dict = defaultdict(list)
        for time, neuron in zip(ESP.t[positions] / ms, ESP.i[positions]):
            spike_dict[time].append(neuron)

        # Находим самую большую группу одновременно спайкующих нейронов
        max_group = max(spike_dict.values(), key=len)
        max_time = max(spike_dict, key=lambda k: len(spike_dict[k]))

        # Формируем данные для построения графика для нейронов из самой большой группы
        max_group_spike_times = []
        max_group_neuron_indices = []
        for time, neurons in spike_dict.items():
            for neuron in neurons:
                if abs(time - max_time) <= 0.25 or neuron in max_group:
                    max_group_spike_times.append(time)
                    max_group_neuron_indices.append(neuron)

        scatter(max_group_spike_times, max_group_neuron_indices, color='red', label='Самая большая группа')
        ylabel('Neuron index')
        legend()
        savefig(path)
        close();

    @staticmethod
    def plot_same_groups(ESP, path_grahp, dt, path_txt=None):

        '''Этот метод строит график групп нейронов, спайкующих одновременно, и сохраняет данные в текстовый файл.

        Параметры:

        ESP: Монитор спайков возбудительных нейронов.
        path_grahp: Путь для сохранения графика.
        dt: Временной интервал для группировки спайков.
        path_txt(Optional(str), default:None): Путь для сохранения данных в текстовый файл.
        '''

        spike_times = [round(spike_time, 2) for spike_time in ESP.t / ms if spike_time <= ESP.t[0] / ms + 100]
        neuron_indices = [neuron for neuron, time in zip(ESP.i, ESP.t / ms) if time <= ESP.t[0] / ms + 100]

        neuron_spikes = list(zip(neuron_indices, spike_times))

        # Создание групп нейронов на основе условия разницы времени спайков меньше dt
        neuron_groups = []
        current_group = [neuron_spikes[0]]
        prev_neuron, prev_time = neuron_spikes[0]

        for neuron, time in neuron_spikes[1:]:
            if time - prev_time <= dt:
                current_group.append((neuron, time))
            else:
                neuron_groups.append(current_group)
                current_group = [(neuron, time)]
                prev_time = time

        neuron_groups.append(current_group)

        if path_txt:
            with open(path_txt, 'w') as file:
                for item in neuron_groups:
                    file.write(f"{item}\n")

        # Создание цветовой метки для каждой группы нейронов
        colors_2 = ['grey', 'm']
        colors_3 = ['b', 'r']
        colors_10 = ['gold', 'pink']

        # Построение точечного графика с окрашиванием нейронов по группам
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_ylabel('Индексы нейронов')
        ax.set_xlabel('Время спайка')
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True)
        for i, group in enumerate(neuron_groups):

            color_2 = colors_2[i % len(colors_2)]
            color_3 = colors_3[i % len(colors_3)]
            color_10 = colors_10[i % len(colors_10)]

            if len(group) == 1:
                for neuron, time in group:
                    plt.scatter(time, neuron, color='black', s=10, label=f'Нейрон {neuron}')
            if len(group) == 2:
                for neuron, time in group:
                    plt.scatter(time, neuron, color=color_2, s=10, label=f'Нейрон {neuron}')
            if len(group) >= 3 and len(group) < 10:
                for neuron, time in group:
                    plt.scatter(time, neuron, color=color_3, s=10, label=f'Нейрон {neuron}')
            if len(group) > 10:
                for neuron, time in group:
                    plt.scatter(time, neuron, color=color_10, s=10, label=f'Нейрон {neuron}')

        fig.savefig(path_grahp)
        plt.close(fig)

    @staticmethod
    def plot_same_spikes(ESP, path, dt):

        '''
        Этот метод строит график нейронов, спайкующих одновременно, и выделяет нейроны из самой большой группы.

        Параметры
        ESP: Монитор спайков возбудительных нейронов.
        path: Путь для сохранения графика.
        dt: Временной интервал для группировки спайков.
        '''

        plt.rcParams["figure.figsize"] = (20, 6)

        spike_dict = defaultdict(list)
        for time, neuron in zip(ESP.t / ms, ESP.i):
            spike_dict[time].append(neuron)

        # Находим самую большую группу одновременно спайкующих нейронов
        max_group = max(spike_dict.values(), key=len)
        max_time = max(spike_dict, key=lambda k: len(spike_dict[k]))

        # Формируем данные для построения графика для нейронов из самой большой группы
        max_group_spike_times = []
        max_group_neuron_indices = []
        for time, neurons in spike_dict.items():
            for neuron in neurons:
                if abs(time - max_time) <= dt or neuron in max_group:
                    max_group_spike_times.append(time)
                    max_group_neuron_indices.append(neuron)

        scatter(max_group_spike_times, max_group_neuron_indices, color='red', label='Самая большая группа')
        ylabel('Neuron index')
        legend()
        savefig(path)
        close();