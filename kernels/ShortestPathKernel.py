import utils 
import numpy as np
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


class ShortestPath:
    
    # Calculates the kernel between two frequency lists using the shortest walk kernel.
    def shortest_walk_kernel(self, f1, f2):
        K = 0
        for i in f1.keys():
            if(i in f2.keys()):
                K += (f1[i] * f2[i])
        return K
    
    # Returns a string representation of a given path in a graph.
    def to_seq(self, G, path):
        seq = ''
        for i in range(len(path)):
            seq = seq + '#' + str(G.nodes[path[i]]['labels'][0]) + '#' + str(G.nodes[path[i]]['labels'][1]) + '#'
            if(i < len(path) - 1):
                seq = seq + str(G.edges[(path[i],path[i+1])]['labels']) 
        return seq           
    
    # Generates a frequency list for each graph in the given list of graphs using the shortest paths between nodes.
    def generate_freq_walk_list(self, list_graph):
        
        freq_list = []
        for G in tqdm(list_graph):
            
            # Calculates the shortest paths from each node to every other node in the graph.
            path_dict = dict(nx.all_pairs_dijkstra_path(G, weight='labels'))
            freq = dict()
            
            # For each pair of nodes in the graph
            for i in G.nodes:
                for j in G.nodes:
                    if(i not in path_dict):
                        continue
                    if(j not in path_dict[i]):
                        continue
                    # Convert the shortest path between node i and node j to a string sequence representation.
                    seq = self.to_seq(G,path_dict[i][j])
                    if(seq not in freq):
                        freq[seq] = 0
                    freq[seq] += 1
                    
            freq_list.append(freq)
            
        return freq_list
    
    def compute_kernel(self, list_graph_a, list_graph_b):
        
        # get the number of graphs in list_graph_a and list_graph_b
        na = len(list_graph_a)
        nb = len(list_graph_b)
        
        # initialize an empty kernel matrix of size na x nb
        K = np.zeros((na,nb))
    
        # generate frequency walk lists for graphs in list_graph_a and list_graph_b
        freq_list_a = []
        freq_list_b = []  
        freq_list_a = self.generate_freq_walk_list(list_graph_a)
        freq_list_b = self.generate_freq_walk_list(list_graph_b)
            
        # compute the kernel value for each pair of graphs in list_graph_a and list_graph_b
        for i in tqdm(range(na)):
            for j in range(nb):
                # compute the shortest walk kernel between the ith graph in list_graph_a and jth graph in list_graph_b
                k = self.shortest_walk_kernel(freq_list_a[i],freq_list_b[j])
                # compute the shortest walk kernel between the ith graph in list_graph_a and itself
                k1 = self.shortest_walk_kernel(freq_list_a[i],freq_list_a[i])
                # compute the shortest walk kernel between the jth graph in list_graph_b and itself
                k2 = self.shortest_walk_kernel(freq_list_b[j],freq_list_b[j])
                # compute the kernel value between the ith graph in list_graph_a and jth graph in list_graph_b
                K[i][j] = k/(k1+k2- k)
        
        # return the kernel matrix
        return K
