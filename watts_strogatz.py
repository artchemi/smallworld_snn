import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

G = nx.watts_strogatz_graph(10, 5, 0.5)
node_opts = {"node_size": 50, "node_color": "r", "alpha": 0.4}

nx.draw_networkx(G, **node_opts)

print('Graph info: ', G)
print('Synapse: ', G.edges)
print(G.degree)

# Заполнить матрицу связностей по данным о ребрах графа
# Заполнить матрицу размером N на N, где N - число нейронов в скрытом слое

plt.show()
