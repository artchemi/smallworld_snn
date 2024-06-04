import argparse

import layers
from utils import *
from tqdm import tqdm
import pandas as pd
import os
import csv

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

    out, mem, tau_max, syn_matrix = hidden_layer.intra_forward(input_spikes)
    df = pd.DataFrame(np.concatenate((input_spikes, mem, out), axis=1))

    df_name = f'{random_dir_name}/data_{random_dir_name}.csv'
    df_name1 = f'{random_dir_name}/syn_{random_dir_name}.csv'
    df.to_csv(df_name, index=False)
    syn_matrix.to_csv(df_name1, index=False)

    #with open(f'{random_dir_name}/corr_{random_dir_name}.csv', "w", newline="") as file:
        #csv_writer = csv.writer(file)
        #for row in spikes:
            #csv_writer.writerow(row)

    # ---draw heatmap and save image---
    plot_heatmap_2_0(membrane_matrix=mem, inp_matrix=input_spikes, dir_name=random_dir_name)
    # ---draw graph and save image---
    hidden_layer.draw_graph()


if __name__ == '__main__':
    main()
