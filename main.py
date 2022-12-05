import sys,itertools, math
import numpy as np
from graph import *
from kmeans import *

def read_input(inputFile):
    """ Read and parse the input file, returning the list of points and its dimension """
    with open(inputFile, "r") as f:
        type, size = f.readline().strip().split()
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
    return graph

def l1_norm(vector):
    """ Given an 1-dimensional numpy array, returns its L1 norm """
    return np.sum(np.absolute(vector))

def enough_cluster(vec, K):
    input = vec.tolist()
    return len(set(input)) >= K

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
    eigenvalues, eigenvectors = np.linalg.eigh(g.L)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]

    input = eigenvectors[:,0:K]
    print(input)
    program = Kmeans(input, K)
    centroids, index = program.run(50)
    print(index)
    return index
        
def create_graph():
    
    points = []
    sample = np.linspace(0, 2*math.pi, num=100).tolist()
    
    f_x = lambda t, R: R*math.cos(t)
    f_y = lambda t, R: R*math.sin(t)
    
    for t in sample:
        p2 = (f_x(t, 1), f_y(t, 1))
        points.append(p2)
    for t in sample:
        p1 = (f_x(t, 100), f_y(t, 100))
        points.append(p1)
    
    return np.asarray(points)

def sym(x, y, sigma):
    a = np.linalg.norm(x - y)**2
    b = -a/(2*sigma**2)
    return math.exp(b)

def points_to_graph(points):
    size = len(points)
    pts = points
    sigma = 1
    connections = []
    for i in range(0, size):
        for j in range(0, size):
            if i != j:
                connections.append((i, j, sym(pts[i], pts[j], sigma)))
            else:
                connections.append((i, j, 0))
    graph = Graph(size, connections)
    return graph
        

# The main body of the code:
if __name__ == "__main__":
    np.set_printoptions(precision=5)
    input_file = sys.argv[1]
    K = int(sys.argv[2])
    graph = read_input(input_file)
    # Check that graph is un-directed

    # points = create_graph()
    # graph = points_to_graph(points)
    # print(graph.A)
    assert graph.is_undirected(), "Graph must be undirected!"
    # visualize_undirected_graph(graph)
    index = spectral_clustering(graph, K)
    visualize_undirected_graph(graph, index)
    
    