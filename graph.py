import numpy as np
from functools import reduce

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