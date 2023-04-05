import utils 
import numpy as np
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


import networkx as nx
import numpy as np
from tqdm import tqdm

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

class RandomWalkKernel:
    
    def __init__(self, lamda=0.5, n_iterations=0):
        """
        Constructor for RandomWalkKernel class.

        Args:
        - lamda: damping factor for the kernel (default: 0.5)
        - n_iterations: number of iterations to run for computing the kernel (default: 0)
        """
        self.n_iterations = n_iterations
        self.lamda = lamda
        
    def walk_kernel(self, G1, G2):
        """
        Compute the walk kernel between two graphs using a finite number of iterations.

        Args:
        - G1: a dictionary representing the pointer graph for the first graph
        - G2: a dictionary representing the pointer graph for the second graph

        Returns:
        - K: the computed walk kernel value
        """
        # Compute the product graph of G1 and G2
        A = utils.compute_product_graph(G1, G2)
        # If the product graph is empty, return 0
        if A is None:
            return 0
        # Compute the sum of each column of the adjacency matrix of the product graph
        s = np.sum(A, axis=0)
        # Normalize the adjacency matrix by the column sums
        A = A / s[:, None]
        # Initialize the kernel value to 0
        K = 0
        # Get the number of nodes in the product graph
        n = A.shape[0]
        # Run the random walk kernel for n_iterations times
        for i in range(self.n_iterations):
            # Update the kernel value
            K += np.ones(n).T @ A @ np.ones(n)
            # Update the adjacency matrix using the damping factor and the previous matrix
            A = self.lamda * A @ A
            
        return K
    
    def infinity_walk_kernel(self, G1, G2):
        """
        Compute the walk kernel between two graphs using an infinite number of iterations.

        Args:
        - G1: a dictionary representing the pointer graph for the first graph
        - G2: a dictionary representing the pointer graph for the second graph

        Returns:
        - K: the computed walk kernel value
        """
        # Compute the product graph of G1 and G2
        A = utils.compute_product_graph(G1, G2)
        # If the product graph is empty, return 0
        if A is None:
            return 0
        # Compute the sum of each column of the adjacency matrix of the product graph
        s = np.sum(A, axis=0)
        # Normalize the adjacency matrix by the column sums
        A = A / s[:, None]
        # Get the number of nodes in the product graph
        n = A.shape[0]
        # Compute the kernel using the formula for an infinite number of iterations
        K = np.ones(n).T @ np.linalg.inv(1 - self.lamda * A) @ np.ones(n)
        
        return K

    def generate_pointers_graph(self, list_graph):
        # Create an empty list to hold pointers for each graph
        pointers_list = []
        
        # Loop through each graph in the list
        for G in tqdm(list_graph):
            # Create an empty dictionary to hold the pointers for the current graph
            pointers = dict()
            
            # Loop through each edge in the graph
            for e in G.edges:
                # Get the starting and ending node indices for the edge
                a = e[0]
                b = e[1]
                
                # Get the labels for the starting and ending nodes
                l_a = G.nodes[a]['labels'][0]
                l_b = G.nodes[b]['labels'][0]
                
                # Get the edge label
                w = G.edges[e]['labels']
                
                # If the edge label, starting node label, and ending node label 
                # are not already in the pointers dictionary, add them with an empty list
                if((l_a,l_b,w) not in pointers):
                    pointers[(l_a,l_b,w)] = []
                    
                # Add the current edge to the list of pointers for the current label and edge label
                pointers[(l_a,l_b,w)].append((a,b))
            
            # Add the pointers dictionary for the current graph to the pointers list
            pointers_list.append(pointers)
        
        # Return the list of pointers dictionaries
        return pointers_list

    
    def compute_kernel(self, list_graph_a, list_graph_b):
        
        # Get number of graphs in each list
        na = len(list_graph_a)
        nb = len(list_graph_b)
        
        # Initialize kernel matrix
        K = np.zeros((na,nb))
        
        # Generate pointers for each graph in both lists
        pointers_list_a = self.generate_pointers_graph(list_graph_a)
        pointers_list_b = self.generate_pointers_graph(list_graph_b)
          
        # Compute kernel value for each pair of graphs
        for i in tqdm(range(na)):
            for j in range(nb):
                # Initialize kernel value to 0
                k = 0
                # Get pointers for current pair of graphs
                G1 = pointers_list_a[i]
                G2 = pointers_list_b[j]
                
                # Compute kernel value based on number of iterations
                if(self.n_iterations == 0):
                    # Use infinite walk kernel
                    k =  self.infinity_walk_kernel(G1,G2)
                else:
                    # Use regular walk kernel
                    k = self.walk_kernel(G1,G2)
                
                # Add kernel value to matrix
                K[i][j] = k
       
        # Return kernel matrix
        return K

                
        
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

    
class KShortestPath:
    
    def __init__(self, maxK) :
        self.maxK = maxK
        
    
    # Returns a string representation of a given path in a graph.
    def to_seq(self, G, path):
        seq = ''
        for i in range(len(path)):
            seq = seq + '#' + str(G.nodes[path[i]]['labels'][0]) + '#' + str(G.nodes[path[i]]['labels'][1]) + '#'
            if(i < len(path) - 1):
                seq = seq + str(G.edges[(path[i],path[i+1])]['labels']) 
        return seq   
    
    def shortest_kernel(self, f1, f2):
        K = 0
        for i in f1.keys():
            if(i in f2.keys()):
                K += (f1[i] * f2[i])
        return K
    
    
    
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
                    
                    iter = 1
                    
                    for v in nx.shortest_simple_paths(G,i,j, weight='labels'):
                        if(iter > self.maxK):
                            break
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
                k = self.shortest_kernel(freq_list_a[i],freq_list_b[j])
                # compute the shortest walk kernel between the ith graph in list_graph_a and itself
                k1 = self.shortest_kernel(freq_list_a[i],freq_list_a[i])
                # compute the shortest walk kernel between the jth graph in list_graph_b and itself
                k2 = self.shortest_kernel(freq_list_b[j],freq_list_b[j])
                # compute the kernel value between the ith graph in list_graph_a and jth graph in list_graph_b
                K[i][j] = k/(k1+k2- k)
        
        # return the kernel matrix
        return K
   
    
    