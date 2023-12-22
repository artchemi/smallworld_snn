import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('neurons_coord.csv')

print(df['x'].loc[0], type(df['x'].loc[0]))

G = nx.Graph()

G.add_node('Hamburg', pos=(53.5672, 10.0285))
G.add_node('Berlin', pos=(52.51704, 13.38792))

G.add_edge('Hamburg', 'Berlin', weight=1000, width=100)
G.add_edge('Berlin', 'Hamburg', weight=1000, width=100)

print(nx.get_node_attributes(G, 'pos'))
print(G.edges)

nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=100)

plt.show()
