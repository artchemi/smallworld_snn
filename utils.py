import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import chain


class Decorators:
    def __init__(self) -> None:
        pass

    @staticmethod
    def benchmark(func):
        """
        Декоратор для проверки времени выполнения функции (прохода спайков по малому миру)
        :param func:
        :return:
        """

        def wrapper(*args, **kwargs):
            t_start = time.time()
            func(*args, **kwargs)
            t_end = time.time()
            # Выводить время в секундах
            print(f'Время выполнения: {t_end - t_start}')

        return wrapper

    @staticmethod
    def mem_reset(func):
        """
        Декоратор для сброса мембранного потенциала каждого нейрона
        :param func:
        :return:
        """

        def wrapper(*args, **kwargs):
            func(*args, **kwargs)

        return wrapper

    # Написать декоратор для записи мембранного потенциала каждого нейрона в отдельный файл


def spike_generator(n, steps=1000, version='1') -> np.ndarray:
    """
    Функция для генерации входных спайков, для проверки работоспособности сети
    :param n:
    :param steps:
    :param version:
    :return:
    """
    spikes = []
    match version:
        case '1':
            # ---random spikes for every columns---
            for i in range(0, steps):
                temp_spikes_lst = []
                while len(temp_spikes_lst) < n:
                    temp_spike = 10 if np.random.choice([True, False], 1)[0] else 0
                    temp_spikes_lst.append(temp_spike)

                spikes.append(temp_spikes_lst)

        case '2':
            # ---constant spike series with random columns---
            size = int(np.round((n / 3), 0))
            # size = 1
            index_rand = np.random.choice(a=n, size=size)
            temp_spikes = np.zeros(n)

            for index in index_rand:
                temp_spikes[index] = 10

            spikes = [temp_spikes.tolist()] * steps

        case '3':
            # ---periodical current---
            size = int(np.round((n / 2), 0))
            index_rand = np.random.choice(a=n, size=size)
            spike_series = []

            while len(spike_series) < steps:
                if (len(spike_series) > 100) and (len(spike_series) < 500):
                    spike_series.append(10.0)
                else:
                    spike_series.append(0.0)

            spikes_series = np.array(spike_series)
            spikes = np.zeros(shape=(n, steps))

            for index in index_rand:
                spikes[index] = spikes_series

            spikes = spikes.T

            return spikes

        case '4':
            # ---Puassone---
            pass

        # spikes.append([10, 0, 10, 0, 10, 0, 10, 0, 10, 0])

    return np.array(spikes)


def plot_neurons_demo(df_name: str) -> None:
    df = pd.read_csv(df_name)
    time_step = np.arange(df.shape[0])

    fig, axes = plt.subplots(2, 5, figsize=(12, 8))

    for ax, i in zip(axes.ravel(), range(0, 10)):
        ax.plot(time_step, df[f'neuron_{i}'], color='black')
        ax.plot(time_step, df[f'inp_spike_{i}'], color='#86d9ad')
        ax.plot(time_step, df[f'out_spike_{i}'], color='red')
        ax.plot([time_step[0], time_step[-1]], [30, 30], color='grey', linestyle=':')

        ax.set_ylabel('Membrane potential, mV')
        ax.set_xlabel('Time step, ms')
        ax.set_title(f'Neuron {i}')

    fig.legend(['Membrane potential', 'Input current', 'Output current', 'Threshold'])
    fig.tight_layout()
    plt.show()


def plot_heatmap(df_name: str, dir_name: str) -> None:
    df = pd.read_csv(df_name)
    columns = df.columns

    new_columns_mem = {}
    new_columns_inp = {}
    col_mem_lst = []
    col_inp_lst = []

    for col_name in columns:
        if 'neuron_' in col_name:
            new_col_name = col_name.replace('neuron_', '')
            new_columns_mem[f'{col_name}'] = new_col_name

            col_mem_lst.append(new_col_name)

        elif 'inp_spike_' in col_name:
            new_col_name = col_name.replace('inp_spike_', '')
            new_columns_inp[f'{col_name}'] = new_col_name

            col_inp_lst.append(new_col_name)

        else:
            continue

    # ---new dataframe for membrane potential heatmap---
    df_membrane = df.rename(columns=new_columns_mem)[col_mem_lst].T

    # ---new dataframe for input spikes heatmap---
    df_input = df.rename(columns=new_columns_inp)[col_inp_lst].T

    # ---heatmaps---
    plt.subplot(2, 1, 1)
    sns.heatmap(df_membrane, cmap='plasma')
    plt.xlabel('Time steps, ms')
    plt.ylabel('Neuron`s index')

    plt.subplot(2, 1, 2)
    sns.heatmap(df_input, cmap='Greys')
    plt.xlabel('Time steps, ms')
    plt.ylabel('Neuron`s index')

    plt.tight_layout()
    plt.savefig(f'{dir_name}/heatmap.png', dpi=300, format='PNG')
    plt.show()


def plot3d_potential(df_name: str) -> None:
    df = pd.read_csv(df_name)
    columns = df.columns

    # ---Вынести переопределние столбцов в датафрейме в отдельную функцию
    new_columns_mem = {}
    col_mem_lst = []

    for col_name in columns:
        if 'neuron_' in col_name:
            new_col_name = col_name.replace('neuron_', '')
            new_columns_mem[f'{col_name}'] = new_col_name

            col_mem_lst.append(new_col_name)

    df_membrane = df.rename(columns=new_columns_mem)[col_mem_lst]

    xgrid = []  # indexes
    ygrid = [list(range(0, df_membrane.shape[0]))]  # time steps
    zgrid = []  # membrane potential

    for col in df_membrane.columns:
        # print(col)
        zgrid.append(df_membrane[col].tolist())
        # xgrid.append([float(col)] * df_membrane.shape[0])
        xgrid.append([float(col)])

    # ---unpacking---
    # xgrid = np.asarray(list(chain(*xgrid)))
    ygrid = np.asarray(list(chain(*ygrid)))
    # zgrid = np.asarray(list(chain(*zgrid)))
    zgrid = np.asarray(zgrid)
    print(xgrid[0])
    print(f'X: {len(xgrid)}, Y: {len(ygrid)}, Z: {len(zgrid)}')
    fig = plt.figure(figsize=(7, 4))
    ax_3d = fig.add_subplot(projection='3d')

    # ax_3d.plot_trisurf(xgrid, ygrid, zgrid, linewidth=0, antialiased=False)
    ax_3d.plot_surface(xgrid, ygrid, zgrid, cmap='plasma')

    plt.show()


def plot_shit(df_name: str):
    df = pd.read_csv(df_name)
    columns = df.columns

    new_columns_mem = {}
    col_mem_lst = []

    for col_name in columns:
        if 'neuron_' in col_name:
            new_col_name = col_name.replace('neuron_', '')
            new_columns_mem[f'{col_name}'] = new_col_name

            col_mem_lst.append(new_col_name)

    # ---new dataframe for membrane---
    df_membrane = df.rename(columns=new_columns_mem)[col_mem_lst]
    x = np.arange(df_membrane.shape[0])

    for col in df_membrane.columns:
        y = df_membrane[col]
        plt.plot(x, y)

    plt.xlabel('Time step, ms')
    plt.ylabel('Membrane potential, mV')

    plt.show()


def main():
    df_name = 'hidden_neurons/data_mem.csv'
    # plot_neurons_demo(df_name=df_name)
    # plot_heatmap(df_name)
    # plot_shit(df_name)
    plot3d_potential(df_name)


if __name__ == '__main__':
    main()
