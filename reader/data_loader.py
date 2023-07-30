import torch
import numpy as np
import time
import numpy as np

def get_edge_labels(max_n):
    
    edge_labels = []
    max_seq_length=2*max_n-1
    max_aux = max_n - 2

    edge_labels.append([0, 1, 0] + [0,0] * max_aux)
    edge_labels.append([1, 0, 2] + [3,0] * max_aux)
    edge_labels.append([0, 2, 0] + [0,0] * max_aux)
    for idx in range(max_aux):
        edge_labels.append(
            [0, 3, 0] + [0,0] * idx + [0,4] + [0,0] * (max_aux - idx - 1))
        edge_labels.append(
            [0, 0, 0] + [0,0] * idx + [4,0] + [0,0] * (max_aux - idx - 1))
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length])
    edge_labels=torch.from_numpy(edge_labels)
    
    return edge_labels
    
 



def prepare_EC_info(ins_info, device):
    instance_info=dict()
    instance_info["node_num"]=ins_info['node_num']
    instance_info["rel_num"]=ins_info['rel_num']
    instance_info["type_num"] = ins_info['type_num']
    instance_info["max_n"]=ins_info['max_n']
    return instance_info