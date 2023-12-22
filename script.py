import argparse

import numpy as np

import layers
from utils import *
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(
    description='Основной скрипт для запуска моделирования скрытого слоя в'
                ' импульсной нейронной сети, имеющей топологию small-world'
)

parser.add_argument('n', type=int, default=30, help='Number of neurons')
parser.add_argument('k', type=int, default=10,
                    help='Number of synapses for every neurons.'
                         ' Each node is joined with its k nearest neighbors in a ring topology.')
parser.add_argument('p', type=float, default=0.5, help='The probability of rewiring each edge')
parser.add_argument('layer_type', type=str, default='ws',
                    help='Type of layer: Watts-Strogatz ("ws"), Biological ("bio")')
parser.add_argument('generator_type', type=str, default='1',
                    help='Type of generator, see documentation'
                         ' or utils.spike_generator(n, steps=1000, version="1") -> np.ndarray')
parser.add_argument('steps', type=int, default=1000, help='Num steps for modeling')

# dt
# Добавить больше аргументов и флагов!!!

args = parser.parse_args()


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


def main():
    # ---name for directory with results---
    random_dir_name = make_random_name(10)

    os.makedirs(random_dir_name)

    match args.layer_type:
        case 'ws':
            hidden_layer = layers.IntraConnectLayer(args.n, args.k, args.p, random_dir_name)
            hidden_layer.from_edges()

        case 'bio':
            print('In dev...')
            return None

        case _:
            print('Incorrect layer_type')
            return None

    input_spikes = spike_generator(n=args.n, steps=args.steps, version=args.generator_type)

    for i in tqdm(range(len(input_spikes)), ascii=True, desc='forward'):
        out = hidden_layer.intra_forward(input_spikes[i])

    print('Done!')

    df_name = f'{random_dir_name}/data_mem.csv'
    # ---draw heatmap and save image---
    plot_heatmap(df_name, random_dir_name)
    # ---draw graph and save image---
    hidden_layer.draw_graph()


if __name__ == '__main__':
    main()
