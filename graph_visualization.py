import networkx as nx
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

G_1 = nx.watts_strogatz_graph(1000, 100, 0.5)
G_2 = nx.watts_strogatz_graph(1000, 50, 0.5)
G_3 = nx.watts_strogatz_graph(1000, 10, 0.5)
G_4 = nx.watts_strogatz_graph(1000, 5, 0.5)

labels = ['n=100, k=2, p=0.5', 'n=50, k=2, p=0.5',
          'n=100, k=1, p=0.5', 'n=50, k=1, p=0.5']

node_opts = {"node_size": 50, "node_color": "r", "alpha": 0.4}

for ax, G, label in zip(axes.ravel(), (G_1, G_2, G_3, G_4), labels):
    nx.draw_networkx(G, ax=ax, **node_opts)
    ax.set_title(label)

fig.tight_layout()
plt.show()
