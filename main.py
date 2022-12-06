import sys,itertools, math
import numpy as np
from graph import *
from kmeans import *
from util import *

data_flag = ["s", "e"]
graph_flag = "g"

def read_input(inputFile):
    """ Read and parse the input file, returning the list of points and its dimension """
    with open(inputFile, "r") as f:
        first_line = f.readline().strip().split()
        size, type = first_line[0], first_line[1]
        if type == graph_flag:
            connections = []
            i = 0
            for line in f.readlines():
                tokens = line.strip().split()
                tokens = list(map(lambda x: int(x), tokens))
                neighbor = tokens[::2]
                weight = tokens[1::2]
                for idx, elt in enumerate(neighbor):
                    connections.append([i, elt, weight[idx]])
                i = i + 1
            graph = Graph(int(size), connections)
            return graph, None, type
        elif type in data_flag:
            points = []
            for line in f.readlines():
                tokens = line.strip().split()
                tokens = list(map(lambda x: float(x), tokens))
                points.append(tokens)
            points = np.asarray(points)
            if type == "s":
                # Similarity Function
                sigma = float(first_line[2])
                return points_to_graph_s(points, sigma), points, type
            if type == "e":
                # Epsilon Neighborhoods
                epsilon = float(first_line[2])
                return points_to_graph_e(points, epsilon), points, type
        else:
            print("Error: Input file type is not supported.")
            sys.exit(0)

def power_iteration_clustering(g, K):
    graph_laplacian = g.L
    rw_normalized_laplacian = np.matmul(np.linalg.inv(g.D), graph_laplacian)
    # This is the normalized affinity matrix
    # W = np.matmul(np.linalg.inv(g.D), graph_laplacian)# np.eye(g.size) - rw_normalized_laplacian
    W = np.matmul(np.linalg.inv(g.D), g.A)
    epsilon = 1e-5/g.size
    vol_A = np.sum(np.concatenate(g.A))
    v_initial = np.sum(g.A, axis=0)/vol_A
    
    t = 0
    v_list = [v_initial]
    delta_list = [None]
    while True:
        vector = np.matmul(W, v_list[t])
        v_next = vector.flatten()
        v_next = v_next/l1_norm(v_next)
        v_list.append(v_next)
        delta = np.absolute(v_list[t+1] - v_list[t])
        delta_list.append(delta)
        
        if t > 1:
            diff = np.linalg.norm(delta_list[t] - delta_list[t-1])
            if diff < epsilon:
                break
        t = t + 1

    assert enough_cluster(v_list[-1], K)
    v_final = []
    for v in v_list[-1].tolist():
        v_final.append([v])
    v_final = np.asarray(v_final)
    
    program = Kmeans(v_final, K)
    centroids, index = program.run(50)
    print(centroids)
    print(index)

# Naive Implementation of Spectral Clustering - Asks Numpy to find the k eigenvectors   
def spectral_clustering(g, K):
    print(g.D)
    print(g.A)
    print(g.L)
    eigenvalues, eigenvectors = np.linalg.eigh(g.L)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]
    print(eigenvalues)
    input = eigenvectors[:,0:K]
    print(input)
    program = Kmeans(input, K)
    centroids, index = program.run(50)
    print(index)
    return index

def sym(x, y, sigma):
    a = np.linalg.norm(x - y)**2
    b = -a/(2*sigma**2)
    return math.exp(b)

def points_to_graph_s(points, sigma):
    size = len(points)
    pts = points
    connections = []
    for i in range(0, size):
        for j in range(0, size):
            if i != j:
                connections.append((i, j, sym(pts[i], pts[j], sigma)))
            else:
                connections.append((i, j, 0))
    graph = Graph(size, connections)
    return graph

def points_to_graph_e(points, epsilon):
    size = len(points)
    pts = points
    connections = []
    for i in range(0, size):
        for j in range(0, size):
            if i != j:
                if np.linalg.norm(pts[i] - pts[j]) < epsilon:
                    connections.append((i, j, 1))
            else:
                connections.append((i, j, 0))
    graph = Graph(size, connections)
    return graph

# The main body of the code:
if __name__ == "__main__":
    np.set_printoptions(precision=5)
    input_file = sys.argv[1]
    K = int(sys.argv[2])
    graph, points, type = read_input(input_file)
    # Check that graph is un-directed
    assert graph.is_undirected(), "Graph must be undirected!"
    
    # create_graph()
    
    if type == graph_flag:
        visualize_undirected_graph(graph)
        index = spectral_clustering(graph, K)
        visualize_undirected_graph(graph, index)
    else:
        index = spectral_clustering(graph, K)
        # Generating visualization for 2d data
        if len(points[0] == 2):
            visualize_data(points, title="Raw Data")
            visualize_data(points, index=index, title="Spectral Clustering")

    
    