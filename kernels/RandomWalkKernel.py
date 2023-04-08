import utils 
import numpy as np
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

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