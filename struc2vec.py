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
            count = len(queue)
            if self.opt1_reduce_len:
                degree_list = {}
            else:
                degree_list = []
            while count>0:
                top = queue.popleft()
                node = self.idx2node[top]
                degree = len(self.graph[node])

                if self.opt1_reduce_len:
                    degree_list[degree] = degree_list.get(degree,0)+1
                else: degree_list.append(degree)
                for nei in self.graph[node]:
                    nei_idx = node2idx[nei]
                    if not visited[nei_idx]:
                        visited[nei_idx]=True
                        queue.append(nei_idx)
                count -=1
            if self.opt1_reduce_len:
                ordered_degree_list = [(degree,freq) for degree,freq in degree_list.items()]
                ordered_degree_list.sort(key=lambda x: x[0])
            else:
                orderd_degree_list = sorted(degree_list)
            ordered_degree_sequence_dict[level] = orderd_degree_list
            level += 1

        return ordered_degree_sequence_dict


    def calc_structural_distance(self,max_num_layers,workers=1,verbose=0):
        if os.path.exists(self.temp_path+'structural_dist.pkl'):
            structural_dist = pd.read_pickle(self.temp_path+'structural_dist.pkl')
        else:
            if self.opt1_reduce_len:
                dist_func = cost_max 
            else:
                dist_func = cost
            if os.path.exists(self.temp+'degreelist.pkl'):
                degreeList = pd.read_pickle(self.temp_path+'degreelist.pkl')
            else:
                degreelist = self.compute_ordered_degreelist(max_num_layers)
                pd.to_pickle(degreelist,self.temp_path+'degreelist.pkl')

            if self.opt2_reduce_sim_calc:
                degrees = self.create_vector()
                degreeListSelected = {}
                vertices = {}
                n_nodes = len(self.idx)
                for v in self.idx:
                    nbs = get_vertices(v,len(self.graph[self.idx2node[v]]),degrees,n_nodes)
                    vertices = [nbs]
    
    def create_vector(self):
        degrees = {}
        sorted_degrees = set()
        for v in self.idx:
            degree = self.graph[self.idx2node[v]]
            sorted_degrees.add(degree)
            if degree not in degrees:
                degrees[degree] = {}
                degrees[degree][vertices] = []
            degrees[degree][vertices].append(v)
        sorted_degrees = np.array(list(sorted_degrees),dtype='int')
        sorted_degrees = np.sort(sorted_degrees)

        length = len(sorted_degrees)
        for index,val in enumerate(sorted_degrees):
            if index>0:
                degrees[index]['before'] = sorted_degrees[index-1]
            if index < length-1:
                degrees[index]['after'] = sorted_degrees[index+1]
        
        return degrees 


def get_vertices(v,degree_v,degrees,n_nodes):
    a_vertices_selected = 2*math.log(n_nodes,2)
    vertices = []
    try:
        c_v = 0 
        for v2 in degree[degree_v]['vertices']:
            if (v !=v2):
                vertices.append(v2)
                c_v +=1
                if (c_v>a_vertices_selected):
                    raise StopIteration
        if ('before' not in degrees[degree_v]):
            degree_b = -1
        else: degree_b = degrees[degree_v]['before']
            
        if ('after' not in degrees[degree_v]):
            degree_a = -1 
        else: degree_a = degrees[degree_v]['after']

        if (degree_a == -1 and degree_b == -1):
            raise StopIteration 

        degree_now = verifyDegrees(degrees, degree_v, degree_a, degree_b)

def verifyDegrees(degrees,degree_v,degree_a,degree_b):
    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now



def cost(a,b):
    ep = 0.5 
    m = max(a,b)+ep
    mi = min(a,b)+ep 
    return ((m/mi)-1)

def cost_min(a,b):
    ep = 0.5 
    m = max(a[0],b[0])+ep
    mi = min(a[0],b[0])+ep
    return ((m/mi)-1)*min(a[1],b[1])

def cost_max(a,b):
    ep = 0.5 
    m = max(a[0],b[0])+ep
    mi = min(a[0],b[0])+ep
    return ((m/mi)-1)*max(a[1],b[1])