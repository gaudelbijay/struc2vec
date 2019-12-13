import os 
import math 
import shutils 
import collections
import numpy as np 
import pandas as pd 


class Struc2Vec():
    def __init__(self, graph, walk_length=10, num_walks=100, workers=1, verbose=0, stay_prob=0.3, opt1_reduce_len=True, 
                    opt2_reduce_sim_calc=True, opt3_num_layers=None, temp_path='./temp_struc2vec/', reuse=False):
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.idx = list(range(len(self.idx2node)))

        self.opt1_reduce_len = opt1_reduce_len
        self.opt2_reduce_sim_calc = opt2_reduce_sim_calc
        self.opt3_num_layers = opt3_num_layers
        self.reuse = reuse
        self.temp_path = temp_path 
        
        
        self._embeddings = {}

    def calc_structural_distance(self,max_num_layers,workers=1,verbose=0):
        if os.path.exists(self.temp_path+'structural_dist.pkl'):
            structural_dist = pd.read_pickle(self.temp_path+'structural_dist.pkl')
        else:
            if self.opt1_reduce_len:
                dist_func = cost_max 
            else:
                dist_func = cost
    
    def compute_ordered_degreelist(self,max_num_layers):
        degreelist = {}
        vertices = self.idx 
        for v in vertices:
            degreelist[v] = self.get_ordered_degreelist_node(v,max_num_layers)
        return degreelist

    def get_ordered_degreelist_node(self,root,max_num_layers):
        if max_num_layers is None:
            max_num_layers = float('inf')
        
        order_degree_sequence_dict = {}
        visited = [False]*len(self.graph.nodes())
        queue = collections.dqueue()
        level = 0 
        queue.append(root)
        visited[root] = True 

        while(len(queue)>0 and level <= max_num_layers):
            
