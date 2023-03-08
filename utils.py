import numpy as np
import networkx as nx

def compute_product_graph(G1,G2):
    
    '''
    P = nx.tensor_product(G1, G2)
    
    #remove the nodes that they don't share same label
    for n in list(P.nodes):
        if(P.nodes[n]['labels'][0][0] != P.nodes[n]['labels'][1][0]):
            P.remove_node(n)
            
    #remove the edges that they don't share same label        
    for n in list(P.edges):
        if(P.edges[n]['labels'][0][0] != P.edges[n]['labels'][1][0]):
            P.remove_edge(n[0],n[1])
  
    if(len(list(P.nodes)) == 0):
        return None
    '''
    
    '''
    P = nx.Graph()
    nodes = []
    for l in nx.product.product(G1, G2):
        if (G1.nodes[l[0]]['labels'][0] == G2.nodes[l[1]]['labels'][0]):
            nodes.append(l)
    P.add_nodes_from(nodes)
    
    
    
    
    
    edges = []
    for e1 in list(G1.edges):
        for e2 in list(G2.edges):
            a = e1[0]
            b = e1[1]
            c = e2[0]
            d = e2[1]
            if((G1.nodes[a]['labels'][0] == G2.nodes[c]['labels'][0]) and (G1.nodes[b]['labels'][0] == G2.nodes[d]['labels'][0]) and (G1.edges[e1]['labels'] == G2.edges[e2]['labels'])):
                edges.append(((a,c),(b,d)))
    P.add_edges_from(edges)     
    ''' 
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
         
    A = nx.adjacency_matrix(P).A #+ np.eye(len(list(P.nodes)))
    
    
    return A

def compute_product_graph_without_labels(G1,G2):
    
    A1 = nx.adjacency_matrix(G1).A
    A2 = nx.adjacency_matrix(G2).A
         
    A = np.kron(A1,A2)
    A = A + np.eye(A.shape[0])
    
    return A