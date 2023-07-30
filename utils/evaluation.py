import time
import numpy as np
import torch
torch.set_printoptions(precision=8)

def batch_evaluation(batch_results, all_facts, gt_dict,ret_ranks,ent_types,rel_num,ent_num,device):
    for i, result in enumerate(batch_results):
        mask_type = all_facts[4][i]
        if mask_type == 1:
            target = ent_types[all_facts[3][i]-rel_num-3][1:]
            target = torch.LongTensor(target).to(device)
            target = target[torch.nonzero(target).squeeze(1)]
            pos = all_facts[2][i]
            id_fact=all_facts[0][i]
            keys=list()
            for j in range(id_fact.size(0)):
                if j ==pos :
                    continue
                if j % 2 ==0 :
                    if id_fact[j] ==0:
                        keys.append(str(id_fact[j].item()))
                    else:
                        keys.append(' '.join (str(x) for x in ent_types[id_fact[j]-3-rel_num][1:]))
                else:
                    keys.append(str(id_fact[j].item()))
            key=" ".join(keys[x] for x in range(len(keys)))
           

            rm_idx = torch.LongTensor(list(gt_dict[pos.item()][key])).to(device)
            rm_idx_tmp = torch.empty(0).to(device)
            for j in rm_idx:
                if j not in target:
                    rm_idx_tmp = torch.cat([rm_idx_tmp,j.unsqueeze(0)],0)
            rm_idx = rm_idx_tmp.long()

            if rm_idx.shape != torch.Size([0]):
                result.index_fill_(0,rm_idx-ent_num-3,-np.Inf)
            
            sortidx = torch.argsort(result,dim=-1,descending=True)
            target = target - ent_num - 3
            ap, ndcg, p_1, p_5, p_10, r_1, r_5, r_10 = compute_metrics_new(sortidx, target)
            ap = torch.tensor(ap).float().unsqueeze(0).to(device)
            ret_ranks['entity_ap']=torch.cat([ret_ranks['entity_ap'],ap],dim=0)
            ndcg = torch.tensor(ndcg).float().unsqueeze(0).to(device)
            ret_ranks['entity_ndcg']=torch.cat([ret_ranks['entity_ndcg'],ndcg],dim=0)
            p_1 = torch.tensor(p_1).float().unsqueeze(0).to(device)
            ret_ranks['entity_p_1']=torch.cat([ret_ranks['entity_p_1'],p_1],dim=0)
            p_5 = torch.tensor(p_5).float().unsqueeze(0).to(device)
            ret_ranks['entity_p_5']=torch.cat([ret_ranks['entity_p_5'],p_5],dim=0)
            p_10 = torch.tensor(p_10).float().unsqueeze(0).to(device)
            ret_ranks['entity_p_10']=torch.cat([ret_ranks['entity_p_10'],p_10],dim=0)
            r_1 = torch.tensor(r_1).float().unsqueeze(0).to(device)
            ret_ranks['entity_r_1'] = torch.cat([ret_ranks['entity_r_1'], r_1], dim=0)
            r_5 = torch.tensor(r_5).float().unsqueeze(0).to(device)
            ret_ranks['entity_r_5'] = torch.cat([ret_ranks['entity_r_5'], r_5], dim=0)
            r_10 = torch.tensor(r_10).float().unsqueeze(0).to(device)
            ret_ranks['entity_r_10'] = torch.cat([ret_ranks['entity_r_10'], r_10], dim=0)
            if (pos == 0) or (pos == 2):
                ret_ranks['ht_ap'] = torch.cat([ret_ranks['ht_ap'], ap], dim=0)
                ret_ranks['ht_ndcg'] = torch.cat([ret_ranks['ht_ndcg'], ndcg], dim=0)
                ret_ranks['ht_p_1'] = torch.cat([ret_ranks['ht_p_1'], p_1], dim=0)
                ret_ranks['ht_p_5'] = torch.cat([ret_ranks['ht_p_5'], p_5], dim=0)
                ret_ranks['ht_p_10'] = torch.cat([ret_ranks['ht_p_10'], p_10], dim=0)
                ret_ranks['ht_r_1'] = torch.cat([ret_ranks['ht_r_1'], r_1], dim=0)
                ret_ranks['ht_r_5'] = torch.cat([ret_ranks['ht_r_5'], r_5], dim=0)
                ret_ranks['ht_r_10'] = torch.cat([ret_ranks['ht_r_10'], r_10], dim=0)
            else:
                ret_ranks['v_ap'] = torch.cat([ret_ranks['v_ap'], ap], dim=0)
                ret_ranks['v_ndcg'] = torch.cat([ret_ranks['v_ndcg'], ndcg], dim=0)
                ret_ranks['v_p_1'] = torch.cat([ret_ranks['v_p_1'], p_1], dim=0)
                ret_ranks['v_p_5'] = torch.cat([ret_ranks['v_p_5'], p_5], dim=0)
                ret_ranks['v_p_10'] = torch.cat([ret_ranks['v_p_10'], p_10], dim=0)
                ret_ranks['v_r_1'] = torch.cat([ret_ranks['v_r_1'], r_1], dim=0)
                ret_ranks['v_r_5'] = torch.cat([ret_ranks['v_r_5'], r_5], dim=0)
                ret_ranks['v_r_10'] = torch.cat([ret_ranks['v_r_10'], r_10], dim=0)



        else:
            target = all_facts[3][i]
            pos = all_facts[2][i]
            id_fact=all_facts[0][i]
            keys=list()
            for j in range(id_fact.size(0)):
                if j ==pos :
                    continue
                if j % 2 ==0 :
                    if id_fact[j] ==0:
                        keys.append(str(id_fact[j].item()))
                    else:
                        keys.append(' '.join (str(x) for x in ent_types[id_fact[j]-3-rel_num][1:]))
                else:
                    keys.append(str(id_fact[j].item()))
            key=" ".join(keys[x] for x in range(len(keys)))

            rm_idx = torch.LongTensor(gt_dict[pos.item()][key]).to(device)
            rm_idx=torch.where(rm_idx!=target,rm_idx,1)
            result.index_fill_(0,rm_idx-3,-np.Inf)
            sortidx = torch.argsort(result,dim=-1,descending=True)
            target = target - 3
            target = torch.LongTensor([target]).to(device)
            ap, ndcg, p_1, p_5, p_10, r_1, r_5, r_10 = compute_metrics_new(sortidx, target)
            ap = torch.tensor(ap).float().unsqueeze(0).to(device)
            ret_ranks['relation_ap'] = torch.cat([ret_ranks['relation_ap'], ap], dim=0)
            ndcg = torch.tensor(ndcg).float().unsqueeze(0).to(device)
            ret_ranks['relation_ndcg'] = torch.cat([ret_ranks['relation_ndcg'], ndcg], dim=0)
            p_1 = torch.tensor(p_1).float().unsqueeze(0).to(device)
            ret_ranks['relation_p_1'] = torch.cat([ret_ranks['relation_p_1'], p_1], dim=0)
            p_5 = torch.tensor(p_5).float().unsqueeze(0).to(device)
            ret_ranks['relation_p_5'] = torch.cat([ret_ranks['relation_p_5'], p_5], dim=0)
            p_10 = torch.tensor(p_10).float().unsqueeze(0).to(device)
            ret_ranks['relation_p_10'] = torch.cat([ret_ranks['relation_p_10'], p_10], dim=0)
            r_1 = torch.tensor(r_1).float().unsqueeze(0).to(device)
            ret_ranks['relation_r_1'] = torch.cat([ret_ranks['relation_r_1'], r_1], dim=0)
            r_5 = torch.tensor(r_5).float().unsqueeze(0).to(device)
            ret_ranks['relation_r_5'] = torch.cat([ret_ranks['relation_r_5'], r_5], dim=0)
            r_10 = torch.tensor(r_10).float().unsqueeze(0).to(device)
            ret_ranks['relation_r_10'] = torch.cat([ret_ranks['relation_r_10'], r_10], dim=0)
            if pos == 1:
                ret_ranks['r_ap'] = torch.cat([ret_ranks['r_ap'], ap], dim=0)
                ret_ranks['r_ndcg'] = torch.cat([ret_ranks['r_ndcg'], ndcg], dim=0)
                ret_ranks['r_p_1'] = torch.cat([ret_ranks['r_p_1'], p_1], dim=0)
                ret_ranks['r_p_5'] = torch.cat([ret_ranks['r_p_5'], p_5], dim=0)
                ret_ranks['r_p_10'] = torch.cat([ret_ranks['r_p_10'], p_10], dim=0)
                ret_ranks['r_r_1'] = torch.cat([ret_ranks['r_r_1'], r_1], dim=0)
                ret_ranks['r_r_5'] = torch.cat([ret_ranks['r_r_5'], r_5], dim=0)
                ret_ranks['r_r_10'] = torch.cat([ret_ranks['r_r_10'], r_10], dim=0)
            else:
                ret_ranks['k_ap'] = torch.cat([ret_ranks['k_ap'], ap], dim=0)
                ret_ranks['k_ndcg'] = torch.cat([ret_ranks['k_ndcg'], ndcg], dim=0)
                ret_ranks['k_p_1'] = torch.cat([ret_ranks['k_p_1'], p_1], dim=0)
                ret_ranks['k_p_5'] = torch.cat([ret_ranks['k_p_5'], p_5], dim=0)
                ret_ranks['k_p_10'] = torch.cat([ret_ranks['k_p_10'], p_10], dim=0)
                ret_ranks['k_r_1'] = torch.cat([ret_ranks['k_r_1'], r_1], dim=0)
                ret_ranks['k_r_5'] = torch.cat([ret_ranks['k_r_5'], r_5], dim=0)
                ret_ranks['k_r_10'] = torch.cat([ret_ranks['k_r_10'], r_10], dim=0)


    return ret_ranks

def compute_metrics_new(ranked_list, ground_truth):

    # AP
    hits = 0
    sum_precs = 0
    ranked_list = ranked_list.cpu().detach().tolist()
    ground_truth = ground_truth.cpu().detach().tolist()
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n+1.0)
    if hits > 0:
        AP = sum_precs / len(ground_truth)
    else:
        AP = 0

    # precision & recall @ k
    predict_1 = [ranked_list[0]]
    predict_5 = ranked_list[:5]
    predict_10 = ranked_list[:10]

    intersection_1 = len(set(predict_1)&set(ground_truth))
    intersection_5 = len(set(predict_5) & set(ground_truth))
    intersection_10 = len(set(predict_10) & set(ground_truth))

    precision_1 = intersection_1 / len(predict_1)
    precision_5 = intersection_5 / len(predict_5)
    precision_10 = intersection_10 / len(predict_10)

    recall_1 = intersection_1 / len(ground_truth)
    recall_5 = intersection_5 / len(ground_truth)
    recall_10 = intersection_10 / len(ground_truth)

    # NDCG
    score = 0.0
    for rank, item in enumerate(ranked_list):
        if item in ground_truth:
            grade = 1.0
            score += grade / np.log2(rank + 2)

    norm = 0.0
    for rank in range(len(ground_truth)):
        grade = 1.0
        norm += grade / np.log2(rank + 2)

    ndcg = score / norm

    return AP, ndcg, precision_1, precision_5, precision_10, recall_1, recall_5, recall_10