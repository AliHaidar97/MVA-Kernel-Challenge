import numpy as np
import networkx as nx
import copy

def fix_graphs(train_graphs, test_graphs):
    
    for G in train_graphs:
        for e in G.nodes:
            G.nodes[e]['labels'] = [G.nodes[e]['labels'][0], 1]

    for G in test_graphs:
        for e in G.nodes:
            G.nodes[e]['labels'] = [G.nodes[e]['labels'][0], 1]

    for G in train_graphs:
        for e in G.edges:
            G.edges[e]['labels'] = G.edges[e]['labels'][0] + 1

    for G in test_graphs:
        for e in G.edges:
            G.edges[e]['labels'] = G.edges[e]['labels'][0] + 1
            
    return train_graphs, test_graphs



def morgan_index(graphs):
    
    for (i,G) in enumerate(graphs):
        K = copy.deepcopy(G)
        for node in G.nodes:
            K.nodes[node]['labels'][1] = 0
            for x in G.neighbors(node):
                K.nodes[node]['labels'][1]  += G.nodes[x]['labels'][1]
        graphs[i] = K 

    return graphs

def compute_product_graph(G1,G2):
    
    P = nx.Graph()
    edges = []
    for l in G1.keys():
        if(l in G2):
            list_edges_1 = G1[l]
            list_edges_2 = G2[l]
            for (a,b) in list_edges_1:
                for (c,d) in list_edges_2:
                    edges.append(((a,c),(b,d)))
                    
    P.add_edges_from(edges)
    
    if(len(list(P.nodes))==0):
        return None
         
    A = nx.adjacency_matrix(P) 
    
    
    return A

def compute_product_graph_without_labels(G1,G2):
    
    A1 = nx.adjacency_matrix(G1).A
    A2 = nx.adjacency_matrix(G2).A
         
    A = np.kron(A1,A2)
    A = A + np.eye(A.shape[0])
    
    return A