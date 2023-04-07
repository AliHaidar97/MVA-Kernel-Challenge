import numpy as np
import networkx as nx

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
         
    A = nx.adjacency_matrix(P) #+ np.eye(len(list(P.nodes)))
    
    
    return A

def compute_product_graph_without_labels(G1,G2):
    
    A1 = nx.adjacency_matrix(G1).A
    A2 = nx.adjacency_matrix(G2).A
         
    A = np.kron(A1,A2)
    A = A + np.eye(A.shape[0])
    
    return A