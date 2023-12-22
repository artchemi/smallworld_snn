import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

name_df = 'neurons_coord.csv'


def generate_node_coord(n: int) -> None:
    """
    Generate coordinate for n neurons
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

    df_coord.to_csv(name_df, index=False)


def calculate_dist(x_1: float, y_1: float, x_2: float, y_2: float) -> float:
    """
    Calculation of distance between 2 nodes in 2-dimensions surface
    :param x_1:
    :param y_1:
    :param x_2:
    :param y_2:
    :return:
    """

    d = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)

    return np.round(d, 2)


def synapse_connect(graph, node_1, node_2, d: float, p_near: float, p_far: float, radius=10):
    """

    :param d:
    :param p:
    :param radius:
    :return:
    """
    # near area
    if d <= radius and np.random.choice([True, False], 1, p=[p_near, 1 - p_near])[0]:
        syn_force = 5.0
        graph.add_edge(node_1, node_2, weight=syn_force, capacity=0)

    # far area
    elif d > radius and np.random.choice([True, False], 1, p=[p_far, 1 - p_far])[0]:
        syn_force = 1.0
        graph.add_edge(node_1, node_2, weight=syn_force, capacity=0)


def neuron_pairs(n: int) -> list:
    neurons_index = np.arange(0, n)

    meshgrid = np.array(np.meshgrid(neurons_index, neurons_index)).T.reshape(-1, 2).tolist()
    pairs_lst = meshgrid.copy()

    for pair in meshgrid:
        if pair[0] == pair[1] or pair[1] < pair[0]:
            pairs_lst.remove(pair)
        else:
            continue

    return pairs_lst


def draw_graph(color='r') -> None:
    """
    Draw graph from random coordinates
    :param color: (='r') color of nodes
    :return: None
    """
    node_opts = {"node_color": "r", "alpha": 0.4}

    G = nx.Graph()
    df_coord = pd.read_csv(name_df)
    pos_dict = {}

    for i in range(0, df_coord.shape[0]):
        x_pos = df_coord['x'].loc[i]
        y_pos = df_coord['y'].loc[i]

        pos_dict[f'{i}'] = (x_pos, y_pos)

        G.add_node(f'{i}', pos=(x_pos, y_pos), size=200)

    pairs = neuron_pairs(df_coord.shape[0])

    for pair in pairs:
        x_1 = df_coord['x'].loc[pair[0]]
        y_1 = df_coord['y'].loc[pair[0]]
        x_2 = df_coord['x'].loc[pair[1]]
        y_2 = df_coord['y'].loc[pair[1]]

        distance = calculate_dist(x_1, y_1, x_2, y_2)
        print('Distance: ', distance)

        syn_opts = {'p_near': 0.9, 'p_far': 0.05, 'radius': 250}

        synapse_connect(G, f'{pair[0]}', f'{pair[1]}', distance, **syn_opts)

    print(G)
    print('Neurons', nx.get_node_attributes(G, 'pos'))
    print('Synapse: ', nx.get_edge_attributes(G, 'weight'))
    nx.draw_networkx(G, pos=pos_dict, with_labels=True, **node_opts)

    #nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=1)

    plt.title(f'P_near = {syn_opts["p_near"]}, P_far = {syn_opts["p_far"]}, R = {syn_opts["radius"]}')
    plt.show()


def main():
    neurons = 100
    generate_node_coord(neurons)

    df = pd.read_csv(name_df)
    print(df)

    draw_graph()


if __name__ == '__main__':
    main()
