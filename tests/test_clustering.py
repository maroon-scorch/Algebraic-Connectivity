from main import *
from graph import Graph
import numpy as np
import sys, math

sys.path.insert(1, '../')

def test_two_clusters():
    points = []
    param = np.linspace(0, 2*math.pi, num=50)
    for t in param:
        points.append((math.cos(t), math.sin(t)))
    for t in param:
        points.append((10*math.cos(t), 10*math.sin(t)))
    points = np.asarray(points)

    graph = points_to_graph_e(points, 1.5)
    index = spectral_clustering(graph, 2)

    # Checks the clusters are exactly the first 50 points and the second 50 points
    assert len(set(index[0:49])) == 1
    assert len(set(index[50:])) == 1
        
def test_disconnected():
    # This is a graph with three connected componets
    connections = []
    connections.append((0, 1, 1))
    connections.append((1, 0, 1))
    connections.append((2, 3, 1))
    connections.append((3, 2, 1))
    connections.append((4, 5, 1))
    connections.append((5, 4, 1))

    graph = Graph(6, connections)
    index = spectral_clustering(graph, 3)
    # Checks the clusters are exactly the disjoint components
    assert len(set(index[0:1])) == 1
    assert len(set(index[2:3])) == 1
    assert len(set(index[4:5])) == 1
    
def test_connected():
    # This is a graph with a triangle and a point
    connections = []
    connections.append((0, 1, 1))
    connections.append((1, 0, 1))
    connections.append((0, 2, 1))
    connections.append((2, 0, 1))
    connections.append((1, 2, 1))
    connections.append((2, 1, 1))
    connections.append((1, 3, 1))
    connections.append((3, 1, 1))

    graph = Graph(4, connections)
    index = spectral_clustering(graph, 2)
    assert len(set(index[0:2])) == 1
    assert len(set(index[3:])) == 1 
    
    # Same graph but with different weight
    connections = []
    connections.append((0, 1, 1))
    connections.append((1, 0, 1))
    connections.append((0, 2, 1))
    connections.append((2, 0, 1))
    connections.append((1, 2, 1))
    connections.append((2, 1, 1))
    connections.append((1, 3, 100))
    connections.append((3, 1, 100))
    
    graph = Graph(4, connections)
    index = spectral_clustering(graph, 2)
    assert len(set([index[0], index[2]])) == 1
    assert len(set([index[1], index[3]])) == 1 