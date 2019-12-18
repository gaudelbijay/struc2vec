def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    for i,node in enumerate(graph.nodes()):
        node2idx[node] = i 
        idx2node.append(node)
    return idx2node,node2idx 

def partition_dict(vertices,workers):
    batch_size = (len(vertices)-1)//workers +1 
    part_list = []
    part = []
    count = 0 
    for v,nbs in vertices.items():
        part.append((v,nbs))
        count +=1 
        if count%batch_size == 0:
            part_list.append(part)
            part=[]
    if len(part)>0:
        part_list.append(part)
    return part_list
