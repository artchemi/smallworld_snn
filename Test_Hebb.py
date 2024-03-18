import argparse

import layers
from utils import *
from tqdm import tqdm
import pandas as pd
import os

parser = argparse.ArgumentParser(
    description='Основной скрипт для запуска моделирования скрытого слоя в'
                ' импульсной нейронной сети, имеющей топологию small-world'
)

parser.add_argument('n', type=int, default=10, help='Number of neurons')
parser.add_argument('k', type=int, default=10,
                    help='Number of synapses for every neurons.'
                         ' Each node is joined with its k nearest neighbors in a ring topology.')
parser.add_argument('p', type=float, default=0.5, help='The probability of rewiring each edge')
parser.add_argument('generator_type', type=str, default='1',
                    help='Type of generator, see documentation'
                         ' or utils.spike_generator(n, steps=1000, version="1") -> np.ndarray')
parser.add_argument('steps', type=int, default=1000, help='Num steps for modeling')
parser.add_argument('dt', type=float, default=0.1, help='Integral step')

# dt
# Добавить больше аргументов и флагов!!!

args = parser.parse_args()


def main():
    # ---name for directory with results---
    random_dir_name = make_random_name(10)
    # ---creating dataframe for input spikes, membrane potential and output spikes---
    #df = create_df(args.n)

    os.makedirs(random_dir_name)

    hidden_layer = layers.IntraConnectLayer(args.n, args.k, args.p, args.dt, random_dir_name)
    hidden_layer.from_edges()

    input_spikes = spike_generator(n=args.n, steps=args.steps, version=args.generator_type)

    print('Saved to folder: ', random_dir_name)

    out, mem = hidden_layer.intra_forward(input_spikes)
    df = pd.DataFrame(np.concatenate((input_spikes, mem, out), axis=1))

    last_n_columns = df.iloc[:, -args.n:]
    indexes = last_n_columns.apply(find_index_of_spikes, args=(10.0,))
    indexes.dropna(inplace=True)
    max_len = max(map(len, indexes))
    filled_lists = [list(filter(None, x)) + [np.nan] * (max_len - len(x)) for x in indexes]

    tau_max = find_tau_max(np.array(filled_lists), args.dt)
    print(f'Tau maximum: {tau_max}')

    df_name = f'{random_dir_name}/data_{random_dir_name}.csv'
    df.to_csv(df_name, index=False)

    # ---draw heatmap and save image---
    plot_heatmap(df_name, random_dir_name)
    # ---draw graph and save image---
    hidden_layer.draw_graph()


if __name__ == '__main__':
    main()
