import utils 
import numpy as np
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


class WalkKernel:
    
    def __init__(self, maxK):
        # maxK is the maximum depth of the BFS search
        self.maxK = maxK
        
    def walk_kernel(self, f1, f2):
        # Calculates the Walk kernel between two graphs represented as frequency lists
        K = 0
        for i in f1.keys():
            if(i in f2.keys()):
                K+= (f1[i][0] * f2[i][0])
        return K

    def generate_path(self, G, bfs_dic, node, seq, size, paths):
        # Recursively generates all possible paths of maximum depth maxK starting from a given node in the graph
        # seq is the current sequence of node and edge labels
        # size is the current length of the sequence
        # paths is the list of all generated paths
        seq = seq + '#' + str(G.nodes[node]['labels'][0]) +  '#' +  str(G.nodes[node]['labels'][1]) + '#'
        size+= 1
        paths.append((seq, size))
        if(node not in bfs_dic.keys()):
            # If the current node is a leaf node, return
            return 
        for i in bfs_dic[node]:
            # For each successor of the current node, append the edge label to the sequence and generate paths starting from that node
            seq_n = seq  + str(G.edges[(node,i)]['labels']) 
            self.generate_path(G,bfs_dic, i, seq_n, size, paths)
    
    
    def generate_freq_list(self, list_graph):
        # Generates the frequency lists for a list of graphs
        freq_list = []
        for G in list_graph:
            freq = dict()
            for node in G.nodes:
                # For each node in the graph, generate all paths starting from that node and add them to the frequency list
                bfs_dic =  dict(nx.bfs_successors(G, node, depth_limit = self.maxK))
                paths = []
                self.generate_path(G, bfs_dic, node, '', 0, paths)
                for p in paths:
                    if(p[0] not in freq):
                        freq[p] = [0, p[1]]
                    freq[p][0] += 1
            freq_list.append(freq)
        return freq_list
        
    def compute_kernel(self, list_graph_a, list_graph_b):
        # Computes the Walk kernel matrix between two lists of graphs
        na = len(list_graph_a)
        nb = len(list_graph_b)
        K = np.zeros((na,nb))
        
        freq_list_a = []
        freq_list_b = []

        # Generate frequency lists for both sets of graphs
        freq_list_a = self.generate_freq_list(list_graph_a)
        freq_list_b = self.generate_freq_list(list_graph_b)
        
        # Compute Walk kernel for each pair of graphs and store in K
        for i in tqdm(range(na)):
            for j in range(nb):
                k = self.walk_kernel(freq_list_a[i],freq_list_b[j])
                k1 = self.walk_kernel(freq_list_a[i],freq_list_a[i])
                k2 = self.walk_kernel(freq_list_b[j],freq_list_b[j])
                K[i][j] = k/(k1+k2- k)
                
        return K