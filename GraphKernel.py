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

    def generate_pointers_graph(self,list_graph):
        pointers_list = []
        for G in tqdm(list_graph):
            pointers = dict()
            for e in G.edges:
                a = e[0]
                b = e[1]
                l_a = G.nodes[a]['labels'][0]
                l_b = G.nodes[b]['labels'][0]
                w = G.edges[e]['labels']
                if((l_a,l_b,w) not in pointers):
                    pointers[(l_a,l_b,w) ] = []
                pointers[(l_a,l_b,w)].append((a,b))
                
            pointers_list.append(pointers)
        
        return pointers_list
    
    def compute_kernel(self, list_graph_a, list_graph_b):
        
        na = len(list_graph_a)
        nb = len(list_graph_b)
        K = np.zeros((na,nb))
        
        pointers_list_a = self.generate_pointers_graph(list_graph_a)
        pointers_list_b = self.generate_pointers_graph(list_graph_b)
        
        '''
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
        return np.array(results).reshape(na,nb)
        ''' 
        
        for i in tqdm(range(na)):
            for j in range(nb):
                k = 0
                G1 = pointers_list_a[i]
                G2 = pointers_list_b[j]
                
                if(self.n_iterations == 0):
                    k =  self.infinity_walk_kernel(G1,G2)
                else:
                    k = self.walk_kernel(G1,G2)
                
                K[i][j] = k
       
        return K
                
        
class ShortestPath:
    
    def shortest_kernel(self, f1, f2):
    
        K = 0
        for i in f1.keys():
            for d in f1[i]:
                    
                if(i in f2.keys() and d in f2[i]):
                    K+= (f1[i][d] * f2[i][d])
                    
             
         
        return K
    
    def generate_freq_list(self,list_graph, d):
        
        freq_list = []
        for i in tqdm(range(len(list_graph))):
            freq = dict()      
            G1 = list_graph[i]
            d1 = d[i]
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
        
            freq_list.append(freq)
            
        return freq_list
       
    def compute_kernel(self, list_graph_a, list_graph_b):
        
        na = len(list_graph_a)
        nb = len(list_graph_b)
        K = np.zeros((na,nb))
        d_a = []
        d_b = []
        freq_list_a = []
        freq_list_b = []

        for i in tqdm(range(len(list_graph_a))):
            d_a.append(dict(nx.all_pairs_dijkstra_path_length(list_graph_a[i],weight='labels')))
        
        for i in tqdm(range(len(list_graph_b))):
            d_b.append(dict(nx.all_pairs_dijkstra_path_length(list_graph_b[i],weight='labels')))

        
        freq_list_a = self.generate_freq_list(list_graph_a,d_a)
        freq_list_b = self.generate_freq_list(list_graph_b,d_b)
            
       
        for i in tqdm(range(na)):
            for j in range(nb):
                k = self.shortest_kernel(freq_list_a[i],freq_list_b[j])
                K[i][j] = k
        
        return K
    
    
class KShortestPath:
    
    def __init__(self, maxK) :
        self.maxK = maxK
        
    
    def shortest_kernel(self, f1, f2):
    
        K = 0
        for i in f1.keys():
            for d in f1[i]:
                if(i in f2.keys() and d in f2[i]):
                    K+= f1[i][d]*f2[i][d]
         
        return K
    
    def generate_freq_list(self,list_graph, d):
        
        freq_list = []
        
        for i in tqdm(range(len(list_graph))):
            freq = dict()  
            for iter in range(1,self.maxK+1):
                    
                G1 = list_graph[i]
                d1 = d[i][iter]

                for a in G1.nodes:
                    for b in G1.nodes:
                        
                        label_a = G1.nodes[a]['labels'][0]
                        label_b = G1.nodes[b]['labels'][0]
                        if(a not in d1):
                            continue
                        if(b not in d1[a]):
                            continue
                        if((label_a,label_b) not in freq):
                            freq[(label_a,label_b)] = dict()
                            
                        if(d1[a][b] not in freq[(label_a,label_b)]):
                            freq[(label_a,label_b)][d1[a][b]] = 0
                        
                        freq[(label_a,label_b)][d1[a][b]] += 1
        
            freq_list.append(freq)
            
        return freq_list
    
    def compute_k_shortest_paths(self,G):
        
        nodes = G.nodes
        freq = dict()
        for i in range(1,self.maxK+1):
            freq[i] = dict()

        for i in nodes:
            for j in nodes:
                iter = 1
                if(nx.has_path(G,i,j) == False):
                    continue
                for v in nx.shortest_simple_paths(G,i,j,weight='labels'):
                    if(iter > self.maxK):break
                    if(i not in freq[iter]):
                        freq[iter][i] = dict()
                    l = nx.path_weight(G,v,weight='labels')
                    freq[iter][i][j] = l 
                    iter+=1
        return freq

    
    def compute_kernel(self, list_graph_a, list_graph_b):
        
        na = len(list_graph_a)
        nb = len(list_graph_b)
        K = np.zeros((na,nb))
        d_a = []
        d_b = []
        freq_list_a = []
        freq_list_b = []

        for i in tqdm(range(len(list_graph_a))):
            d_a.append(self.compute_k_shortest_paths(list_graph_a[i]))
        
        for i in tqdm(range(len(list_graph_b))):
            d_b.append(self.compute_k_shortest_paths(list_graph_b[i]))

        

        freq_list_a = self.generate_freq_list(list_graph_a,d_a)
        freq_list_b = self.generate_freq_list(list_graph_b,d_b)
            
        for i in tqdm(range(na)):
            for j in range(nb):
                k = self.shortest_kernel(freq_list_a[i],freq_list_b[j])
                K[i][j] = +k
        
        return K