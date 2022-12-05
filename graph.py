import numpy as np
from functools import reduce
import networkx as nx
import matplotlib.pyplot as plt

# Data structure of a simple graph
class Graph:
    # Connections A list of tuples (a, b, w) representing an edge from vertex a to vertex b with weight w
    def __init__(self, size, connections):
        # An integer representing number of vertices
        self.size = size 
        # The graph structure represetned in a dictionary
        self.graph = {key: [] for key in range(0, size)}
        for a, b, w in connections:
            self.graph[a].append((b, w))
        # The degree matrix
        self.D = np.diag([reduce(lambda a,b: a + b[1], self.graph[i], 0) for i in range(0, size)])
        # The adjacency matrix
        A = [[0 for _ in range(0, size)] for _ in range(0, size)]
        for i in range(0, size):
            for b, w in self.graph[i]:
                A[i][b] = w
        self.A = np.asarray(A)
        # The Graph Laplacian:
        self.L = self.D - self.A

    def __repr__(self):
        return str(self.graph)

    def __eq__(self, other):
        if type(other) != Graph:
            return False
        if other.size != self.size:
            return False
        # return approx_equal(self, other)
        return self.graph == other.graph

    def __hash__(self):
      return hash(self.graph)
  
    # Checks if the graph is simple or not
    def is_simple(self):
        for i in range(0, self.size):
            neighbors = list(map(lambda x: x[0], self.graph[i]))
            # Check for loops and multiple edges:
            if i in neighbors or (len(neighbors) != len(set(neighbors))):
                return False
        return True
    # Checks if the graph is undirected
    def is_undirected(self):
        return np.all(np.abs(self.A - np.transpose(self.A)) < 1e-9)

# Visualizes a given undirected graph 
def visualize_undirected_graph(g, index=None):
    G = nx.from_numpy_matrix(g.A, create_using=nx.Graph)
    
    if type(index) != type(None):
        color_dict = [index[i] for i in range(0, g.size)]
    else:
        color_dict = ["tab:red" for i in range(0, g.size)]
    
    pos = nx.spring_layout(G)
    # Drawing the graph
    nx.draw_networkx_nodes(G, pos, node_color=color_dict, node_size=700)
    nx.draw_networkx_edges(G, pos, width=6)

    # Drawing the Labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    # Displaying the graph
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()