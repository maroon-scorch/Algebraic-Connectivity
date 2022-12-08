from graph import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Common utility functions
def visualize_undirected_graph(g, index=None):
    """ Visualizes a given undirected graph """
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

def visualize_data(p, index=None, title=""):
    """ Visualizes 2d data """
    x, y = zip(*p)
    if type(index) == type(None):
        plt.scatter(x, y, color="red", alpha=0.5)
    else:
        plt.scatter(x, y, c=index, alpha=0.5)
    plt.title(title)
    plt.show()

def l1_norm(vector):
    """ Given an 1-dimensional numpy array, returns its L1 norm """
    return np.sum(np.absolute(vector))

def enough_cluster(vec, K):
    """ Checks if the K-means operation can be used """
    input = vec.tolist()
    return len(set(input)) >= K