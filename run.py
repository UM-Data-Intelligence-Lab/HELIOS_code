import argparse
import logging
import time
import random
import torch
import pickle
import numpy as np
from utils.args import print_arguments
from reader.data_reader import read_input
from reader.data_loader import prepare_EC_info, get_edge_labels
from model.init_helios import HELIOS
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from utils.evaluation import batch_evaluation

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())


parser = argparse.ArgumentParser(description='Helios model')
parser.add_argument('--dataset', type=str, default='wd50k',help="")
parser.add_argument('--dim', type=int, default=256,help="")
parser.add_argument('--learning_rate', type=float, default=0.0001,help="")
parser.add_argument('--batch_size', type=int, default=1024,help="")
parser.add_argument('--epochs', type=int, default=101,help="")
parser.add_argument("--use_cuda", type=bool,default=True, help="")
parser.add_argument("--gpu", type=int,default=0, help="")
parser.add_argument('--intermediate_size', type=int, default=512,help="")
parser.add_argument('--self_attention_layers', type=int, default=6,help="")
parser.add_argument('--gat_layers', type=int, default=2,help="")
parser.add_argument('--num_attention_heads', type=int, default=4,help="")
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1,help="")
parser.add_argument('--attention_dropout_prob', type=float, default=0.1,help="")
parser.add_argument('--num_edges', type=int, default=5,help="")
parser.add_argument('--sparsifier', type=int, default=10,help="")
args = parser.parse_args()
args.dataset='../data/'+args.dataset

class EDataset(Dataset.Dataset):
    def __init__(self, data,type_num,device):
        self.data=data
        self.type_num=type_num
        self.device=device
    def __len__(self):
        return len(self.data[0])
    def __getitem__(self,index):
        label=np.zeros(self.type_num, dtype=np.float32)
        label[self.data[6][index]]=1
        return  self.data[0][index],self.data[1][index],self.data[2][index],self.data[3][index],\
                self.data[4][index],self.data[5][index],label



def main(args):
    config = vars(args)
    if args.use_cuda:
        device = torch.device("cuda",args.gpu)
        config["device"]=device
    else:
        device = torch.device("cpu")
        config["device"]="cpu"

    with open(args.dataset + "/dictionaries_and_facts.bin", 'rb') as fin:
        data_info = pickle.load(fin)
    rel_keys2id = data_info['roles_indexes']  
    entities_values2id = data_info['values_indexes'] 
    n_rel_keys = len(rel_keys2id)
    n_entities_values = len(entities_values2id)

    helios_info = read_input(args.dataset, args.sparsifier, entities_values2id)

    instance_info = prepare_EC_info(helios_info, device)  
    edge_labels = get_edge_labels(helios_info['max_n']).to(device)

    type_attn_l2_matrix = torch.ones(helios_info['max_n']-1)
    for i in range(helios_info['max_n']):
        row = np.arange(helios_info['max_n'])
        row = np.delete(row, i)
        type_attn_l2_matrix = np.vstack((type_attn_l2_matrix, row))

    type_attn_l2_matrix = type_attn_l2_matrix[1:]
    type_attn_l2_matrix = torch.tensor(type_attn_l2_matrix, dtype=torch.int64).to(device)

    model = HELIOS(instance_info, config, torch.tensor(helios_info['ent_types']).to(device)).to(device)

    helios_train_facts=list()
    for i,helios_train_fact in enumerate(helios_info['train_facts']):
        if i <6 :
            helios_train_fact=torch.tensor(helios_train_fact).to(device)
        helios_train_facts.append(helios_train_fact)
    train_data_E_reader=EDataset(helios_train_facts,helios_info['type_num']+helios_info['rel_num'],device)
    train_E_pyreader=DataLoader.DataLoader(train_data_E_reader,batch_size=args.batch_size,shuffle=True,drop_last=False)

    helios_valid_facts=list()
    for i,helios_valid_fact in enumerate(helios_info['valid_facts']):
        if i <6 :
            helios_valid_fact=torch.tensor(helios_valid_fact).to(device)
        helios_valid_facts.append(helios_valid_fact)
    valid_data_E_reader=EDataset(helios_valid_facts,helios_info['type_num']+helios_info['rel_num'],device)
    valid_E_pyreader=DataLoader.DataLoader(
        valid_data_E_reader,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)

    helios_test_facts=list()
    for i,helios_test_fact in enumerate(helios_info['test_facts']):
        if i <6 :
            helios_test_fact=torch.tensor(helios_test_fact).to(device)
        helios_test_facts.append(helios_test_fact)
    test_data_E_reader=EDataset(helios_test_facts,helios_info['type_num']+helios_info['rel_num'],device)
    test_E_pyreader=DataLoader.DataLoader(
        test_data_E_reader,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)

    helios_optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(helios_optimizer, step_size=20, gamma=0.9)

    for iteration in range(1, args.epochs):
        logger.info("iteration "+str(iteration))
        model.train()
        helios_epoch_loss = 0
        start = time.time()
        for j,data in enumerate(train_E_pyreader):
            helios_pos = data
          
            helios_optimizer.zero_grad()
            helios_loss,_= model.forward(helios_pos,edge_labels,type_attn_l2_matrix)
            helios_loss.backward()
            helios_optimizer.step()
            helios_epoch_loss += helios_loss

         
            if j%100==0:
                logger.info(str(j)+' , loss: '+str(helios_loss.item()))
        scheduler.step()
        end = time.time()
        t2=round(end - start, 2)
        logger.info("epoch_loss = {:.3f}, time = {:.3f} s , lr= {}".format(helios_epoch_loss, t2,scheduler.get_last_lr()[0]) )

        if iteration % 5 ==0 :        
            model.eval()
            
            
            with torch.no_grad():
                h1E = predict(
                    model=model,
                    helios_test_pyreader=valid_E_pyreader,
                    helios_all_facts=helios_info['all_facts_eval'],
                    edge_labels=edge_labels,
                    type_attn_l2_matrix=type_attn_l2_matrix,
                    max_n = instance_info['max_n'],
                    ent_types = helios_info['ent_types'],
                    rel_num = instance_info['rel_num'],
                    ent_num = helios_info['ent_num'],
                    is_test=False,
                    device=device)

                h2E = predict(
                    model=model,
                    helios_test_pyreader=test_E_pyreader,
                    helios_all_facts=helios_info['all_facts_eval'],
                    edge_labels=edge_labels,
                    type_attn_l2_matrix=type_attn_l2_matrix,
                    max_n = instance_info['max_n'],
                    ent_types = helios_info['ent_types'],
                    rel_num = instance_info['rel_num'],
                    ent_num = helios_info['ent_num'],
                    is_test=True,
                    device=device)
       

    logger.info("stop")

def predict(model, helios_test_pyreader, helios_all_facts, edge_labels, type_attn_l2_matrix, max_n, ent_types, rel_num, ent_num, is_test, device):
    start=time.time()

    step = 0
    helios_ret_ranks=dict()

    helios_ret_ranks['entity_ap']=torch.empty(0).to(device)
    helios_ret_ranks['entity_ndcg'] = torch.empty(0).to(device)
    helios_ret_ranks['entity_p_1'] = torch.empty(0).to(device)
    helios_ret_ranks['entity_p_5'] = torch.empty(0).to(device)
    helios_ret_ranks['entity_p_10'] = torch.empty(0).to(device)
    helios_ret_ranks['entity_r_1'] = torch.empty(0).to(device)
    helios_ret_ranks['entity_r_5'] = torch.empty(0).to(device)
    helios_ret_ranks['entity_r_10'] = torch.empty(0).to(device)

    helios_ret_ranks['ht_ap'] = torch.empty(0).to(device)
    helios_ret_ranks['ht_ndcg'] = torch.empty(0).to(device)
    helios_ret_ranks['ht_p_1'] = torch.empty(0).to(device)
    helios_ret_ranks['ht_p_5'] = torch.empty(0).to(device)
    helios_ret_ranks['ht_p_10'] = torch.empty(0).to(device)
    helios_ret_ranks['ht_r_1'] = torch.empty(0).to(device)
    helios_ret_ranks['ht_r_5'] = torch.empty(0).to(device)
    helios_ret_ranks['ht_r_10'] = torch.empty(0).to(device)

    helios_ret_ranks['v_ap'] = torch.empty(0).to(device)
    helios_ret_ranks['v_ndcg'] = torch.empty(0).to(device)
    helios_ret_ranks['v_p_1'] = torch.empty(0).to(device)
    helios_ret_ranks['v_p_5'] = torch.empty(0).to(device)
    helios_ret_ranks['v_p_10'] = torch.empty(0).to(device)
    helios_ret_ranks['v_r_1'] = torch.empty(0).to(device)
    helios_ret_ranks['v_r_5'] = torch.empty(0).to(device)
    helios_ret_ranks['v_r_10'] = torch.empty(0).to(device)

    helios_ret_ranks['relation_ap']=torch.empty(0).to(device)
    helios_ret_ranks['relation_ndcg']=torch.empty(0).to(device)
    helios_ret_ranks['relation_p_1']=torch.empty(0).to(device)
    helios_ret_ranks['relation_p_5'] = torch.empty(0).to(device)
    helios_ret_ranks['relation_p_10'] = torch.empty(0).to(device)
    helios_ret_ranks['relation_r_1'] = torch.empty(0).to(device)
    helios_ret_ranks['relation_r_5'] = torch.empty(0).to(device)
    helios_ret_ranks['relation_r_10'] = torch.empty(0).to(device)

    helios_ret_ranks['r_ap']=torch.empty(0).to(device)
    helios_ret_ranks['r_ndcg']=torch.empty(0).to(device)
    helios_ret_ranks['r_p_1']=torch.empty(0).to(device)
    helios_ret_ranks['r_p_5'] = torch.empty(0).to(device)
    helios_ret_ranks['r_p_10'] = torch.empty(0).to(device)
    helios_ret_ranks['r_r_1'] = torch.empty(0).to(device)
    helios_ret_ranks['r_r_5'] = torch.empty(0).to(device)
    helios_ret_ranks['r_r_10'] = torch.empty(0).to(device)

    helios_ret_ranks['k_ap'] = torch.empty(0).to(device)
    helios_ret_ranks['k_ndcg'] = torch.empty(0).to(device)
    helios_ret_ranks['k_p_1'] = torch.empty(0).to(device)
    helios_ret_ranks['k_p_5'] = torch.empty(0).to(device)
    helios_ret_ranks['k_p_10'] = torch.empty(0).to(device)
    helios_ret_ranks['k_r_1'] = torch.empty(0).to(device)
    helios_ret_ranks['k_r_5'] = torch.empty(0).to(device)
    helios_ret_ranks['k_r_10'] = torch.empty(0).to(device)

   

    for i, data in enumerate(helios_test_pyreader):

        helios_pos = data
        _,helios_np_fc_out = model.forward(helios_pos,edge_labels,type_attn_l2_matrix)


        helios_pos[0][:,0::2] = torch.where(helios_pos[0][:,0::2] > rel_num + 2, helios_pos[0][:,0::2], helios_pos[0][:,0::2]-rel_num)


        helios_ret_ranks=batch_evaluation(helios_np_fc_out, helios_pos, helios_all_facts, helios_ret_ranks, ent_types, rel_num, ent_num, device)

        step += 1

    
    entity_ap = helios_ret_ranks['entity_ap'].mean().item()
    entity_ndcg = helios_ret_ranks['entity_ndcg'].mean().item()
    entity_p_1 = helios_ret_ranks['entity_p_1'].mean().item()
    entity_p_5 = helios_ret_ranks['entity_p_5'].mean().item()
    entity_p_10 = helios_ret_ranks['entity_p_10'].mean().item()
    entity_r_1 = helios_ret_ranks['entity_r_1'].mean().item()
    entity_r_5 = helios_ret_ranks['entity_r_5'].mean().item()
    entity_r_10 = helios_ret_ranks['entity_r_10'].mean().item()

    ht_ap = helios_ret_ranks['ht_ap'].mean().item()
    ht_ndcg = helios_ret_ranks['ht_ndcg'].mean().item()
    ht_p_1 = helios_ret_ranks['ht_p_1'].mean().item()
    ht_p_5 = helios_ret_ranks['ht_p_5'].mean().item()
    ht_p_10 = helios_ret_ranks['ht_p_10'].mean().item()
    ht_r_1 = helios_ret_ranks['ht_r_1'].mean().item()
    ht_r_5 = helios_ret_ranks['ht_r_5'].mean().item()
    ht_r_10 = helios_ret_ranks['ht_r_10'].mean().item()

    v_ap = helios_ret_ranks['v_ap'].mean().item()
    v_ndcg = helios_ret_ranks['v_ndcg'].mean().item()
    v_p_1 = helios_ret_ranks['v_p_1'].mean().item()
    v_p_5 = helios_ret_ranks['v_p_5'].mean().item()
    v_p_10 = helios_ret_ranks['v_p_10'].mean().item()
    v_r_1 = helios_ret_ranks['v_r_1'].mean().item()
    v_r_5 = helios_ret_ranks['v_r_5'].mean().item()
    v_r_10 = helios_ret_ranks['v_r_10'].mean().item()

    relation_ap = helios_ret_ranks['relation_ap'].mean().item()
    relation_ndcg = helios_ret_ranks['relation_ndcg'].mean().item()
    relation_p_1 = helios_ret_ranks['relation_p_1'].mean().item()
    relation_p_5 = helios_ret_ranks['relation_p_5'].mean().item()
    relation_p_10 = helios_ret_ranks['relation_p_10'].mean().item()
    relation_r_1 = helios_ret_ranks['relation_r_1'].mean().item()
    relation_r_5 = helios_ret_ranks['relation_r_5'].mean().item()
    relation_r_10 = helios_ret_ranks['relation_r_10'].mean().item()

    r_ap = helios_ret_ranks['r_ap'].mean().item()
    r_ndcg = helios_ret_ranks['r_ndcg'].mean().item()
    r_p_1 = helios_ret_ranks['r_p_1'].mean().item()
    r_p_5 = helios_ret_ranks['r_p_5'].mean().item()
    r_p_10 = helios_ret_ranks['r_p_10'].mean().item()
    r_r_1 = helios_ret_ranks['r_r_1'].mean().item()
    r_r_5 = helios_ret_ranks['r_r_5'].mean().item()
    r_r_10 = helios_ret_ranks['r_r_10'].mean().item()

    k_ap = helios_ret_ranks['k_ap'].mean().item()
    k_ndcg = helios_ret_ranks['k_ndcg'].mean().item()
    k_p_1 = helios_ret_ranks['k_p_1'].mean().item()
    k_p_5 = helios_ret_ranks['k_p_5'].mean().item()
    k_p_10 = helios_ret_ranks['k_p_10'].mean().item()
    k_r_1 = helios_ret_ranks['k_r_1'].mean().item()
    k_r_5 = helios_ret_ranks['k_r_5'].mean().item()
    k_r_10 = helios_ret_ranks['k_r_10'].mean().item()


    helios_all_entity = "ENT_TYPE\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        entity_ap,
        entity_ndcg,
        entity_p_1,
        entity_p_5,
        entity_p_10,
        entity_r_1,
        entity_r_5,
        entity_r_10)

    helios_all_relation = "RELATION\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        relation_ap,
        relation_ndcg,
        relation_p_1,
        relation_p_5,
        relation_p_10,
        relation_r_1,
        relation_r_5,
        relation_r_10)

    helios_all_ht = "H/T_TYPE\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ht_ap,
        ht_ndcg,
        ht_p_1,
        ht_p_5,
        ht_p_10,
        ht_r_1,
        ht_r_5,
        ht_r_10)

    helios_all_r = "PRIMARY_R\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        r_ap,
        r_ndcg,
        r_p_1,
        r_p_5,
        r_p_10,
        r_r_1,
        r_r_5,
        r_r_10)

    helios_all_v = "VALUE_TYPE\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        v_ap,
        v_ndcg,
        v_p_1,
        v_p_5,
        v_p_10,
        v_r_1,
        v_r_5,
        v_r_10)

    helios_all_k = "KEY\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        k_ap,
        k_ndcg,
        k_p_1,
        k_p_5,
        k_p_10,
        k_r_1,
        k_r_5,
        k_r_10)



    if is_test:
        option='Evaluation'
    else:
        option='Validation'
    logger.info("\n-------- "+option+" Performance --------\n%s\n%s\n%s\n%s\n%s\n%s\n%s" % (
        "\t".join(["TASK\t", "mAP", "NDCG", "Prec@1", "Prec@5", "Prec@10", "Recall@1", "Recall@5", "Recall@10"]),
        helios_all_ht, helios_all_r, helios_all_v, helios_all_k, helios_all_entity, helios_all_relation))


    end=time.time()
    logger.info("time: "+str(round(end - start, 3))+'s')


    return None

if __name__ == '__main__':
    print_arguments(args)
    main(args)
