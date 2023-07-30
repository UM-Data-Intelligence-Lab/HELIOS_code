import torch
import torch.nn
from model.model import HeliosModel
from model.graph_encoder import truncated_normal
torch.set_printoptions(precision=16)

class HELIOS(torch.nn.Module):
    def __init__(self, info, config, ent_types):
        super(HELIOS, self).__init__()
        self.config = config
        self.node_num = info["node_num"]

        self.node_embeddings = torch.nn.Embedding(self.node_num, self.config['dim'])
        self.node_embeddings.weight.data = truncated_normal(self.node_embeddings.weight.data, std=0.02)

        self.ent_types = ent_types
        self.edge_embedding_k = torch.nn.Embedding(self.config['num_edges'],self.config['dim'] // self.config['num_attention_heads'])
        self.edge_embedding_k.weight.data = truncated_normal(self.edge_embedding_k.weight.data, std=0.02)

        self.edge_embedding_v = torch.nn.Embedding(self.config['num_edges'],self.config['dim'] // self.config['num_attention_heads'])
        self.edge_embedding_v.weight.data = truncated_normal(self.edge_embedding_v.weight.data, std=0.02)


        self.heliosconfig = dict()
        self.heliosconfig['self_attention_layers'] = self.config['self_attention_layers']
        self.heliosconfig['gat_layers'] = self.config['gat_layers']
        self.heliosconfig['num_attention_heads'] = self.config['num_attention_heads']
        self.heliosconfig['hidden_size'] = self.config['dim']
        self.heliosconfig['intermediate_size'] = self.config['intermediate_size']
        self.heliosconfig['hidden_dropout_prob'] = self.config['hidden_dropout_prob']
        self.heliosconfig['attention_dropout_prob'] = self.config['attention_dropout_prob']
        self.heliosconfig['vocab_size'] = self.node_num
        self.heliosconfig['num_relations'] = info['rel_num']
        self.heliosconfig['num_types'] = info['type_num']
        self.heliosconfig['num_edges'] = self.config['num_edges']
        self.heliosconfig['max_arity'] = info['max_n']
        self.heliosconfig['device'] = self.config['device']
        self.heliosconfig['sparsifier'] = self.config['sparsifier']
        self.model = HeliosModel(self.heliosconfig, self.node_embeddings, self.edge_embedding_k,self.edge_embedding_v).to(self.heliosconfig['device'])


    
    def forward(self,data,edge_labels,type_attn_l2_matrix):
        
        input_ids, input_mask, mask_pos, mask_label, mask_type, mask_t_label, groud_truth = data

        loss, sortidx = self.model(input_ids, input_mask, edge_labels, mask_pos, mask_label, mask_type, self.ent_types, groud_truth, type_attn_l2_matrix)
                                             
                                             
        return loss, sortidx