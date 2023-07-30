import logging
import torch
import torch.nn
import copy
from model.graph_encoder import encoder,truncated_normal,GAT_attention

class HeliosModel(torch.nn.Module):
    def __init__(self, config, node_embeddings,edge_embedding_k,edge_embedding_v):
        super(HeliosModel, self).__init__()

        self._n_layer = config['self_attention_layers']
        self._gat_layer=config['gat_layers']
        self._n_head = config['num_attention_heads']
        self._emb_size = config['hidden_size']
        self._intermediate_size = config['intermediate_size']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_dropout_prob']

        self._voc_size = config['vocab_size'] - config['num_types']  
        self._node_num = config['vocab_size']
        self._n_relation = config['num_relations']
        self._n_type = config['num_types']
        self._n_edge = config['num_edges'] 
        self._max_arity = config['max_arity']
        self._max_seq_len = self._max_arity*2-1
        self._sparsifier = config['sparsifier']

        self._device=config["device"]
        self.node_embedding=node_embeddings
        self.layer_norm1=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True)
        
        self.edge_embedding_k=edge_embedding_k       
        self.edge_embedding_v=edge_embedding_v

        self.encoder_model = encoder(
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._intermediate_size,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout)
            
        self.GAT_attention=GAT_attention(
            input_size=self._emb_size,
            sparsifier=self._sparsifier,
            gat_layers=self._gat_layer
        )

        self.GAT_attention_l2=GAT_attention(
            input_size=self._emb_size,
            sparsifier=self._max_arity + self._sparsifier - 1,
            gat_layers=self._gat_layer
        ) 



        self.layer_norm2=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-7,elementwise_affine=True)
        
        self.fc2_bias = torch.nn.init.constant_(torch.nn.parameter.Parameter(torch.Tensor(self._n_relation)), 0.0)
        self.fc3_bias = torch.nn.init.constant_(torch.nn.parameter.Parameter(torch.Tensor(self._n_type)), 0.0) 
        
        self.loss = torch.nn.CrossEntropyLoss()
        


    def forward(self, input_ids, input_mask, edge_labels, mask_pos, mask_label, mask_type, ent_types, groud_truth, type_attn_l2_matrix):

        mask_type_tmp=copy.copy(mask_type)
        emb_out = self.node_embedding(input_ids)   
        emb_out = torch.nn.Dropout(self._prepostprocess_dropout)(self.layer_norm1(emb_out))  

        type_matrix = ent_types[:,1:] 
        type_matrix = torch.cat((torch.tensor([2]*self._sparsifier).unsqueeze(0).to(self._device), type_matrix), 0)
        type_matrix = torch.cat((torch.tensor([1]*self._sparsifier).unsqueeze(0).to(self._device), type_matrix), 0)
        type_matrix = torch.cat((torch.tensor([0]*self._sparsifier).unsqueeze(0).to(self._device), type_matrix), 0)  

        input_type_ids = input_ids[:, 0::2] - self._n_relation  

        input_type_tmp = type_matrix.index_select(0,input_type_ids.view(-1)).view(-1,self._sparsifier)


        h_tmp = torch.sign(input_type_tmp).unsqueeze(2).float()
        pos = torch.sign(input_type_tmp).unsqueeze(2).float()
        h_tmp = torch.matmul(h_tmp,h_tmp.transpose(1,2)).unsqueeze(3)
        h_tmp = 1000000.0*(h_tmp-1.0)
        input_type = self.node_embedding(input_type_tmp) 

        type_attn_output = self.GAT_attention(    
            input=input_type,
            pos_matrix=h_tmp,
            pos=pos)

        type_attn_output = type_attn_output.view(input_ids.size(0),self._max_arity,self._emb_size)       
        type_attn_output = type_attn_output.unsqueeze(1).repeat(1,self._max_arity,1,1)
        type_attn_output = type_attn_output.view(-1,self._max_arity,self._emb_size)  

       
        type_attn_l2_matrix0 = type_attn_l2_matrix.unsqueeze(0).repeat(input_ids.size(0),1,1)
        type_attn_l2_matrix = type_attn_l2_matrix0.unsqueeze(3).repeat(1,1,1,self._emb_size)
        type_attn_l2_matrix = type_attn_l2_matrix.view(-1,self._max_arity-1,self._emb_size) 

        type_attn_output = torch.gather(type_attn_output, 1, type_attn_l2_matrix) 

        
        type_attn_output_tmp = torch.abs(torch.sum(type_attn_output, 2)) 

        type_attn_output = torch.cat((input_type, type_attn_output), 1)  

        input_type_tmp = torch.cat((input_type_tmp, type_attn_output_tmp), 1) 
        h_tmp = torch.sign(input_type_tmp).unsqueeze(2).float()
        pos = torch.sign(input_type_tmp).unsqueeze(2).float()
        h_tmp = torch.matmul(h_tmp,h_tmp.transpose(1,2)).unsqueeze(3)
        h_tmp = 1000000.0*(h_tmp-1.0)

        type_attn_output = self.GAT_attention_l2(    
            input=type_attn_output,
            pos_matrix=h_tmp,
            pos=pos)
                        
        type_attn_output = type_attn_output.view(input_ids.size(0),self._max_arity,self._emb_size)
        
        emb_out[:,0::2,:] = type_attn_output

        mask_1=torch.tensor(1).to(self._device)
        mask_2=torch.tensor(2).to(self._device)

        
        mask_matrix=torch.where(mask_type==1,mask_2,mask_1)

        emb_out[range(input_ids.size(0)),mask_pos,:]=self.node_embedding(mask_matrix)

        edges_key = self.edge_embedding_k(edge_labels)  
        edges_value = self.edge_embedding_v(edge_labels)  
        edge_mask = torch.sign(edge_labels).unsqueeze(2)
        edges_key = torch.mul(edges_key, edge_mask)
        edges_value = torch.mul(edges_value, edge_mask)

        input_mask=input_mask.unsqueeze(2)
        self_attn_mask = torch.matmul(input_mask,input_mask.transpose(1,2))
        self_attn_mask=1000000.0*(self_attn_mask-1.0)
        n_head_self_attn_mask = torch.stack([self_attn_mask] * self._n_head, dim=1)

        _enc_out = self.encoder_model(
            enc_input=emb_out,
            edges_key=edges_key,
            edges_value=edges_value,
            pos_matrix=n_head_self_attn_mask)

        mask_pos=mask_pos.unsqueeze(1)
        mask_pos=mask_pos[:,:,None].expand(-1,-1,self._emb_size) 
        h_masked = torch.gather(input=_enc_out, dim=1, index=mask_pos).reshape([-1, _enc_out.size(-1)])

        h_masked = torch.nn.GELU()(h_masked)
        h_masked = self.layer_norm2(h_masked)

        fc_out1 = torch.nn.functional.linear(h_masked, self.node_embedding.weight[self._voc_size:self._node_num,:], self.fc3_bias) 

        fc_out2 = torch.nn.functional.linear(h_masked, self.node_embedding.weight[3:(3+self._n_relation),:], self.fc2_bias)  

        fc_out = torch.cat((fc_out2,fc_out1),1) 

        relation_indicator = torch.empty(input_ids.size(0), self._n_relation).to(self._device)
        torch.nn.init.constant_(relation_indicator,-1)

        entity_indicator = torch.empty(input_ids.size(0), (self._n_type)).to(self._device)   
        torch.nn.init.constant_(entity_indicator,1)
        type_indicator = torch.cat((relation_indicator, entity_indicator), dim=1).to(self._device)
        mask_type=mask_type.unsqueeze(1)
        type_indicator = torch.mul(type_indicator, mask_type)
        type_indicator=torch.nn.functional.relu(type_indicator)

        fc_out_mask=1000000.0*(type_indicator-1.0)
        fc_out = torch.add(fc_out, fc_out_mask)

        rel_index=torch.where(mask_type_tmp==-1)[0]
        type_index=torch.where(mask_type_tmp==1)[0]

        one_hot_labels=groud_truth.float().to(self._device)
        t_one_hot_labels=one_hot_labels[type_index,self._n_relation:]
        r_one_hot_labels=one_hot_labels[rel_index,:self._n_relation]

        t_one_hot_labels=t_one_hot_labels /torch.sum(t_one_hot_labels,dim=1).unsqueeze(1) 
        r_one_hot_labels=r_one_hot_labels /torch.sum(r_one_hot_labels,dim=1).unsqueeze(1) 

        fc_out2=fc_out2[rel_index,:]
        fc_out1=fc_out1[type_index,:]

        loss_r=self.loss(fc_out2,r_one_hot_labels)        
        loss_t = self.loss(fc_out1,t_one_hot_labels) 
        loss=(loss_r+loss_t)
        
        return loss, fc_out


