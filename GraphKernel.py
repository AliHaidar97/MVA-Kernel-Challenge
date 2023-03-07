import utils 
import numpy as np
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


class RandomWalkKernel:
    
    def __init__(self, lamda = 0.5, n_iterations = 0):
        self.n_iterations = n_iterations
        self.lamda = lamda
        
    def walk_kernel(self, G1, G2):
        
        A = utils.compute_product_graph(G1,G2)
        if(A is None):
            return 0
        s = np.sum(A,axis = 0)
        A = A/s[:,None]
        K = 0
        n = A.shape[0]
    
        for i in range(self.n_iterations):
            K+= np.ones(n).T@A@np.ones(n)
            A = A@A
            
        return K
    
    def infinity_walk_kernel(self, G1, G2):
        
        A = utils.compute_product_graph(G1,G2)
        if(A is None):
            return 0
        s = np.sum(A,axis = 0)
        A = A/s[:,None]
        n = A.shape[0]
        K = np.ones(n).T@np.linalg.inv(1-self.lamda*A)@np.ones(n)
        
        return K

    def compute_kernel(self, list_graph_a, list_graph_b):
        
        na = len(list_graph_a)
        nb = len(list_graph_b)
        K = np.zeros((na,nb))
        
        
        def kernel(i,j):
            k = 0
            G1 = list_graph_a[i]
            G2 = list_graph_b[j]
            
            if(self.n_iterations == 0):
                k =  self.infinity_walk_kernel(G1,G2)
            else:
                k = self.walk_kernel(G1,G2)
                
            return k 
        
        results = Parallel(n_jobs=-1)(delayed(kernel)(i,j) for i in range(na) for j in range(nb))
        
        '''
        for i in tqdm(range(na)):
            for j in range(nb):
                k = 0
                G1 = list_graph_a[i]
                G2 = list_graph_b[j]
                
                if(self.n_iterations == 0):
                    k =  self.infinity_walk_kernel(G1,G2)
                else:
                    k = self.walk_kernel(G1,G2)
                
                K[i][j] = k
        ''' 
        return np.array(results).reshape(na,nb)
                
        
class ShortestPath:
    
    
    def shortest_kernel(self, f1, f2):
    
        K = 0
        for i in f1.keys():
            for d in f1[i]:
                if(i in f2.keys() and d in f2[i]):
                    K+= f1[i][d]*f2[i][d]
         
        return K
            
    def compute_kernel(self, list_graph_a, list_graph_b):
        
        na = len(list_graph_a)
        nb = len(list_graph_b)
        K = np.zeros((na,nb))
        d_a = []
        d_b = []
        freq_list_a = []
        freq_list_b = []

        for i in tqdm(range(len(list_graph_a))):
            d_a.append(dict(nx.all_pairs_shortest_path_length(list_graph_a[i])))
        
        for i in tqdm(range(len(list_graph_b))):
            d_b.append(dict(nx.all_pairs_shortest_path_length(list_graph_b[i])))
        
        for i in tqdm(range(len(list_graph_a))):
            freq = dict()      
            G1 = list_graph_a[i]
            d1 = d_a[i]
            for a in G1.nodes:
                for b in G1.nodes:
                    
                    label_a = G1.nodes[a]['labels'][0]
                    label_b = G1.nodes[b]['labels'][0]
                    if(b not in d1[a]):
                        continue
                    if((label_a,label_b) not in freq):
                        freq[(label_a,label_b)] = dict()
                        
                    if(d1[a][b] not in freq[(label_a,label_b)]):
                        freq[(label_a,label_b)][d1[a][b]] = 0
                    
                    freq[(label_a,label_b)][d1[a][b]] += 1
        
            freq_list_a.append(freq)
            
        for i in tqdm(range(len(list_graph_b))):
            freq = dict()      
            G1 = list_graph_b[i]
            d1 = d_b[i]
            for a in G1.nodes:
                for b in G1.nodes:
                    
                    label_a = G1.nodes[a]['labels'][0]
                    label_b = G1.nodes[b]['labels'][0]
                    if(b not in d1[a]):
                        continue
                    if((label_a,label_b) not in freq):
                        freq[(label_a,label_b)] = dict()
                        
                    if(d1[a][b] not in freq[(label_a,label_b)]):
                        freq[(label_a,label_b)][d1[a][b]] = 0
                    
                    freq[(label_a,label_b)][d1[a][b]] += 1
        
            freq_list_b.append(freq)
        
        def kernel(i,j):
            k = 0
            G1 = list_graph_a[i]
            G2 = list_graph_b[j]
            d1 = d_a[i]
            d2 = d_b[j]
            k = self.shortest_kernel(G1,G2,d1,d2)
                
            return k 
        
        #results = Parallel(n_jobs=-1)(delayed(kernel)(i,j) for i in range(na) for j in range(nb))
        
       
        for i in tqdm(range(na)):
            for j in range(nb):
                k = self.shortest_kernel(freq_list_a[i],freq_list_b[j])
                K[i][j] = k
        
        return K#np.array(results).reshape(na,nb)