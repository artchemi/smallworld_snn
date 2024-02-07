import argparse

import layers
from utils import *
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(
    description='Скрипт для запуска моделирования скрытого слоя с топологией Watts-Shtrogatz в'
                ' импульсной нейронной сети, имеющей топологию small-world')

parser.add_argument('n', type=int, default=30, help='Number of neurons')
parser.add_argument('r', type=float, default=250, help='Radius of near area')
parser.add_argument('steps', type=int, default=1000, help='Num steps for modeling')
parser.add_argument('dt', type=float, default=0.1, help='Integral step')

args = parser.parse_args()


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


def main():
    random_dir_name = make_random_name(10)

    os.mkdir(random_dir_name)
    df = create_df(args.n)

    hidden_layer = layers.IntraConnectLayerBio(args.n, args.r, 0.9,
                                               0.2, random_dir_name)

    hidden_layer.from_edges()

    input_spikes = spike_generator(args.n, args.steps, '2')

    for i in tqdm(range(len(input_spikes)), ascii=True, desc='forward'):
        out, mem = hidden_layer.intra_forward(input_spikes[i])
        df.loc[len(df.index)] = input_spikes[i].tolist() + mem + out.tolist()

    df_name = f'{random_dir_name}/data_{random_dir_name}.csv'
    df.to_csv(df_name, index=False)

    # ---draw heatmap and save image---
    plot_heatmap(df_name, random_dir_name)
    # ---draw graph and save image---
    hidden_layer.draw_graph()


if __name__ == '__main__':
    main()
