import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import chain
from tqdm import tqdm


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
            for i in tqdm(range(0, steps)):
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


def make_random_name(size: int) -> str:
    name = []

    while len(name) < size:
        if np.random.choice([True, False], 1)[0]:
            # ---char in unicode---
            random_char = chr(np.random.randint(97, 122))

        else:
            # ---digits in unicode---
            random_char = chr(np.random.randint(48, 57))

        name.append(random_char)

    np.random.shuffle(name)
    random_name = ''.join(name)

    return random_name


def create_df(n: int) -> pd.DataFrame:
    columns_spike = []
    columns_neuron = []
    columns_out_spike = []
    for i in range(n):
        columns_spike.append(f'inp_spike_{i}')
        columns_neuron.append(f'neuron_{i}')
        columns_out_spike.append(f'out_spike_{i}')

    df = pd.DataFrame(columns=columns_spike+columns_neuron+columns_out_spike)

    return df


def find_index_of_spikes(column, spike_value):
    index = column[column == spike_value].index
    return np.array(index) if not index.empty else np.nan


def find_tau_max(matrix, dt):

    count = 0
    sum_diff = 0
    for col in range(matrix.shape[1]):
        for i in range(len(matrix[:, col])):
            if not np.isnan(matrix[i][col]):
                for j in range(i + 1, len(matrix[:, col])):
                    if not np.isnan(matrix[j][col]):
                        sum_diff += abs(matrix[i][col] - matrix[j][col])
                        count += 1
    if count != 0:
        return (sum_diff / count) * 2 * dt
    else:
        return 'No spikes'

def find_indices_above_diagonal(matrix, value):
    indixes = np.argwhere(np.triu(matrix, k=1) == value)
    return indixes

def find_indices_above_diagonal(matrix, value):
    indixes = np.argwhere(np.triu(matrix, k=1) == value)
    return indixes

def main():
    df_name = 'hidden_neurons/data_mem.csv'
    # plot_neurons_demo(df_name=df_name)
    # plot_heatmap(df_name)
    # plot_shit(df_name)


if __name__ == '__main__':
    main()
