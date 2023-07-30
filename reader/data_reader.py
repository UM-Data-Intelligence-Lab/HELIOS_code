import collections
import json
import copy
from collections import Counter
import logging
import numpy as np
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())

def build_entity2types_dictionaries(dataset_name, entities_values2id):
    entityName2entityTypes = {}
    entityId2entityTypes = {}
    entityType2entityNames = {}
    entityType2entityIds = {}

    entity2type_file = open(dataset_name, "r")

    for line in entity2type_file:
        splitted_line = line.strip().split("\t")
        entity_name = splitted_line[0][8:]
        entity_type = splitted_line[1][6:]

        if entity_name not in entityName2entityTypes:
            entityName2entityTypes[entity_name] = []
        if entity_type not in entityName2entityTypes[entity_name]:
            entityName2entityTypes[entity_name].append(entity_type)

        if entity_type not in entityType2entityNames:
            entityType2entityNames[entity_type] = []
        if entity_name not in entityType2entityNames[entity_type]:
            entityType2entityNames[entity_type].append(entity_name)

        entity_id = entities_values2id[entity_name]
        if entity_id not in entityId2entityTypes:
            entityId2entityTypes[entity_id] = []
        if entity_type not in entityId2entityTypes[entity_id]:
            entityId2entityTypes[entity_id].append(entity_type)

        if entity_type not in entityType2entityIds:
            entityType2entityIds[entity_type] = []
        if entity_id not in entityType2entityIds[entity_type]:
            entityType2entityIds[entity_type].append(entity_id)

    entity2type_file.close()

    return entityName2entityTypes, entityId2entityTypes, entityType2entityNames, entityType2entityIds


def build_type2id_v2(inputData):
    type2id = {}
    id2type = {}
    type_counter = 0
    with open(inputData) as entity2type_file:
        for line in entity2type_file:
            splitted_line = line.strip().split("\t")
            entity_type = splitted_line[1][6:]

            if entity_type not in type2id:
                type2id[entity_type] = type_counter
                id2type[type_counter] = entity_type
                type_counter += 1

    entity2type_file.close()
    return type2id, id2type


def build_typeId2frequency(dataset_name, type2id):
    if "type2relation2type_ttv" in dataset_name:
        typeId2frequency = {}
        type_relation_type_file = open(dataset_name, "r")

        for line in type_relation_type_file:
            splitted_line = line.strip().split("\t")
            head_type = splitted_line[0][6:]
            tail_type = splitted_line[2][6:]
            head_type_id = type2id[head_type]
            tail_type_id = type2id[tail_type]

            if head_type_id not in typeId2frequency:
                typeId2frequency[head_type_id] = 0
            if tail_type_id not in typeId2frequency:
                typeId2frequency[tail_type_id] = 0

            typeId2frequency[head_type_id] += 1
            typeId2frequency[tail_type_id] += 1

        type_relation_type_file.close()
    elif "type2relation2type2key2type_ttv" in dataset_name:
        typeId2frequency = {}
        type_relation_type_file = open(dataset_name, "r")
        for line in type_relation_type_file:
            splitted_line = line.strip().split("\t")
            for i in range(0, len(splitted_line), 2):
                value_type = splitted_line[i][6:]
                value_type_id = type2id[value_type]
                if value_type_id not in typeId2frequency:
                    typeId2frequency[value_type_id] = 0
                typeId2frequency[value_type_id] += 1
        type_relation_type_file.close()
    return typeId2frequency


def build_entityId2SparsifierType(entities_values2id, type2id, entityId2entityTypes, sparsifier, typeId2frequency,
                                  unk_entity_type_id):
    entityId2SparsifierType = {}
    if sparsifier > 0:
        for i in entities_values2id:
            entityId = entities_values2id[i]
            if entityId in entityId2entityTypes:
                entityTypes = entityId2entityTypes[entityId]
                entityTypeIds = []
                for j in entityTypes:
                    entityTypeIds.append(type2id[j])

                current_freq = {}
                for typeId in entityTypeIds:
                    current_freq[typeId] = typeId2frequency[typeId]

                sorted_current_freq = sorted(current_freq.items(), key=lambda kv: kv[1],
                                             reverse=True)[:sparsifier]
                topNvalueTypes = [item[0] for item in sorted_current_freq]
                entityId2SparsifierType[entityId] = topNvalueTypes

            else:
                entityId2SparsifierType[entityId] = [unk_entity_type_id]
        return entityId2SparsifierType
    else:
        logger.info("SPARSIFIER ERROR!")


def read_facts_new(file, entityName2SparsifierType):
    facts_list = list()
    max_n = 0
    entity_list = list()
    relation_list = list()
    type_list = list()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            fact = list()
            obj = json.loads(line)
            if obj['N'] > 7:
                continue
            if obj['N'] > max_n:
                max_n = obj['N']
            flag = 0
            for key in obj:
                if flag == 0:
                    fact.append(obj[key][0])
                    fact.append(key)
                    fact.append(obj[key][1])
                    relation_list.append(key)
                    entity_list.append(obj[key][0])
                    entity_list.append(obj[key][1])
                    break

            if obj['N'] > 2:
                for kv in obj.keys():
                    if kv != 'N' and kv != key:
                        if isinstance(obj[kv], list):
                            for item in obj[kv]:
                                fact.append(kv)
                                fact.append(item)
                                relation_list.append(kv)
                                entity_list.append(item)
                        else:
                            fact.append(kv)
                            fact.append(obj[kv])
                            relation_list.append(kv)
                            entity_list.append(obj[kv])
            values = fact[0::2]
            for i in values:
                types = entityName2SparsifierType[i]
                type_list.extend(types)
            facts_list.append(fact)
    return facts_list, max_n, relation_list, entity_list, type_list

def read_dict_new(e_ls, r_ls, t_ls):
    dict_id = dict()
    dict_id['PAD'] = 0
    dict_id['E_MASK'] = 1
    dict_id['T_MASK'] = 2
    dict_num = 3

    rel_num = 0
    ent_num = 0
    type_num = 0

    for item in r_ls:
        dict_id[item] = dict_num
        dict_num += 1
        rel_num += 1

    for item in e_ls:
        dict_id[item] = dict_num
        dict_num += 1
        ent_num += 1

    for item in t_ls:
        dict_id[item] = dict_num
        dict_num += 1
        type_num += 1

    return dict_id, dict_num, rel_num, ent_num, type_num


def facts_to_id(facts, max_n, node_dict, rel_num, ent_num, ent_types, type_num):  
    mask_labels = list()
    mask_pos = list()
    mask_types = list()
    id_t_facts = list()
    id_t_masks = list()
    mask_t_labels = list()    

    for fact in facts:
        id_fact = list()
        id_mask = list()

        for i, item in enumerate(fact):
            id_fact.append(node_dict[item])
            id_mask.append(1.0)

        max_fact_length = 2 * max_n - 1
        arity = (len(id_fact) + 1) // 2
        for j, mask_label in enumerate(id_fact):
            x = copy.copy(id_fact)
            y = copy.copy(id_mask)
            if j % 2 == 0:
                x[j] = 2 + rel_num
                mask_type = 1
                mask_t_label = mask_label
            else:
                x[j] = 1
                mask_type = -1
                mask_t_label = rel_num

            x = x + [0, rel_num] * (max_n - arity) 
            y = y + [0] * (max_n - arity) * 2

            id_t_facts.append(x)
            id_t_masks.append(y)
            mask_pos.append(j)  
            mask_labels.append(mask_label)  
            mask_types.append(mask_type)  
            mask_t_labels.append(mask_t_label)
  
    return [id_t_facts, id_t_masks, mask_pos, mask_labels, mask_types, mask_t_labels]


def update(train_facts,train_ground_truth,train_ground_truth_keys,train_max_type_num):
    [id_t_facts, id_t_masks, mask_pos, mask_labels, mask_types, mask_t_labels]=train_facts
    groud_truth=list()
    for i in range(len(mask_pos)):
        tmp=list()
        tmp=train_ground_truth[mask_pos[i]][train_ground_truth_keys[i]]
        assert len(tmp) >0
        groud_truth.append(tmp)
    return [id_t_facts, id_t_masks, mask_pos, mask_labels, mask_types, mask_t_labels, groud_truth]
def get_truth_eval_new(all_facts,max_n,node_dict,ent_types,sparsifier,num_rel):
    max_aux=max_n-2
    max_seq_length = 2 * max_aux + 3
    gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    all_fact_ids=list()
    max_len=0
    for fact in all_facts:
        id_fact=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item])
        all_fact_id=copy.copy(id_fact)
        all_fact_ids.append(all_fact_id)
        while len(id_fact)<(2*max_n-1):
            id_fact.append(0)
        for pos in range(max_seq_length):
            if id_fact[pos]==0:
                continue
            keys=list()
            for j in range(max_seq_length):
                if j ==pos :
                    continue
                if j % 2 ==0 :
                    if id_fact[j] ==0:
                        keys.append(str(id_fact[j]))
                    else:
                        keys.append(' '.join (str(x) for x in ent_types[id_fact[j]-3-num_rel][1:]))
                else:
                    keys.append(str(id_fact[j]))
            key=" ".join(keys[x] for x in range(len(keys)))
            if pos %2 ==0:
                value=set(ent_types[id_fact[pos]-3-num_rel][1:])
                value.discard(0)
                gt_dict[pos][key]=list(set(gt_dict[pos][key])|value)
                if(len(gt_dict[pos][key])>max_len):
                    max_len=len(gt_dict[pos][key])
            else :
                gt_dict[pos][key].append(id_fact[pos])
    return gt_dict,all_fact_ids


def get_ground_truth(train_facts,max_n,node_dict,ent_types,sparsifier, rel_num, ent_num, type_num):
    max_aux=max_n-2
    max_seq_length = 2 * max_aux + 3
    gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    all_fact_ids=list()
    ground_truth_keys=list()
    max_len=0
    for fact in train_facts:
        id_fact=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item])
        all_fact_id=copy.copy(id_fact)
        all_fact_ids.append(all_fact_id)
        while len(id_fact)<(2*max_n-1):
            id_fact.append(0)
        for pos in range(max_seq_length):
            if id_fact[pos]==0:
                continue
            keys=list()
            for j in range(max_seq_length):
                if j ==pos :
                    continue
                if j % 2 ==0 :
                    if id_fact[j] ==0:
                        keys.append(str(id_fact[j]))
                    else:
                        keys.append(' '.join (str(x) for x in ent_types[id_fact[j]-3-rel_num][1:])) 
                else:
                    keys.append(str(id_fact[j]))
            key=" ".join(keys[x] for x in range(len(keys)))
            ground_truth_keys.append(key)
            if pos %2 ==0:
                value=set(ent_types[id_fact[pos]-3-rel_num][1:])
                value.discard(0)
                value=set(np.array(list(value))-ent_num-3)
                gt_dict[pos][key]=list(set(gt_dict[pos][key])|value)
                if(len(gt_dict[pos][key])>max_len):
                    max_len=len(gt_dict[pos][key])
            else :
                gt_dict[pos][key].append(id_fact[pos]-3)
    return gt_dict,ground_truth_keys,max_len

def helper(train_ground_truth):
    max_len=0
    for key,values in train_ground_truth.items():
        for k,v in values.items():
            train_ground_truth[key][k]=list(set(v))
            if(len(train_ground_truth[key][k])>max_len):
                max_len=len(train_ground_truth[key][k])
    return train_ground_truth
def update_facts(train_facts,valid_facts,test_facts,ent_types,node_dict,rel_num,max_n):
    start=time.time()
    type_valid_test_facts=list()
    trian_facts_new=list()
    removed_train_facts=list()
    max_seq_length = 2 * max_n -1
    for fact in test_facts:
        id_fact=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item])
        while len(id_fact)<(2*max_n-1):
            id_fact.append(0)
        list_tmp=list()
        for i in range(max_seq_length):
            if id_fact[i]==0 or i%2!=0:
                list_tmp.append(id_fact[i])
            else:
                list_tmp.append(ent_types[id_fact[i]-3-rel_num][1:])
        type_valid_test_facts.append(list_tmp)

    for fact in valid_facts:
        id_fact=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item])
        while len(id_fact)<(2*max_n-1):
            id_fact.append(0)
        list_tmp=list()
        for i in range(max_seq_length):
            if id_fact[i]==0 or i%2!=0:
                list_tmp.append(id_fact[i])
            else:
                list_tmp.append(ent_types[id_fact[i]-3-rel_num][1:])
        type_valid_test_facts.append(list_tmp)

    for fact in train_facts:
        id_fact=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item])
        while len(id_fact)<(2*max_n-1):
            id_fact.append(0)
        list_tmp=list()
        for i in range(max_seq_length):
            if id_fact[i]==0 or i%2!=0:
                list_tmp.append(id_fact[i])
            else:
                list_tmp.append(ent_types[id_fact[i]-3-rel_num][1:])
        if list_tmp not in type_valid_test_facts:
            trian_facts_new.append(fact)
        else:
            removed_train_facts.append(fact)
    return trian_facts_new,valid_facts,test_facts



def ent_to_type(e_list, entityName2SparsifierType, node_dict, sparsifier):
    type_ls = list()
    type_pos_ls = list()
    for e in e_list:
        e_types = list()
        type_pos = list()
        e_types.append(node_dict[e])
        e_type = entityName2SparsifierType[e]
        e_type = [node_dict[i] for i in e_type]
        e_types.extend(e_type)
        type_pos.extend([1] * len(e_types))
        while len(e_types) < (sparsifier+1):
            e_types.append(0)
            type_pos.append(0)
        type_ls.append(e_types)
        type_pos_ls.append(type_pos)

    return type_ls, type_pos_ls


def get_input(train_file, valid_file, test_file, file1, file2, file3, sparsifier, entities_values2id):

    entityName2entityTypes, entityId2entityTypes, entityType2entityNames, entityType2entityIds = \
        build_entity2types_dictionaries(file1, entities_values2id)

    type2id, id2type = build_type2id_v2(file1)
    n_types = len(type2id)

    entity_typeId2frequency = build_typeId2frequency(file2, type2id)
    value_typeId2frequency = build_typeId2frequency(file3, type2id)

    entity_typeId2frequency_tmp, value_typeId2frequency_tmp = Counter(entity_typeId2frequency), Counter(
        value_typeId2frequency)
    typeId2frequency = dict(entity_typeId2frequency_tmp + value_typeId2frequency_tmp)

    unk_entity_type_id = len(type2id)

    entityId2SparsifierType = build_entityId2SparsifierType(entities_values2id,
                                                            type2id,
                                                            entityId2entityTypes,
                                                            sparsifier,
                                                            typeId2frequency,
                                                            unk_entity_type_id)

    id2type[n_types] = "unknown"
    type2id["unknown"] = n_types
    e_ls = list(entities_values2id.keys())
    entityName2SparsifierType = dict()
    for key in entityId2SparsifierType.keys():
        entityName2SparsifierType[e_ls[key]] = list()
        for item in entityId2SparsifierType[key]:
            typeName = id2type[item] + '_type'
            entityName2SparsifierType[e_ls[key]].append(typeName)

    train_facts, max_train, train_r, train_e, train_t = read_facts_new(train_file, entityName2SparsifierType) 
    valid_facts, max_valid, valid_r, valid_e, valid_t = read_facts_new(valid_file, entityName2SparsifierType)
    test_facts, max_test, test_r, test_e, test_t = read_facts_new(test_file, entityName2SparsifierType)

    max_n = max(max_train, max_valid, max_test)
    e_list = list(set(train_e + valid_e + test_e))
    r_list = list(set(train_r + valid_r + test_r))
    t_list = list(set(train_t + valid_t + test_t))
    node_dict, node_num, rel_num, ent_num, type_num = read_dict_new(e_list,r_list,t_list)  

    ent_types, _ = ent_to_type(e_list, entityName2SparsifierType, node_dict, sparsifier)
    train_facts,valid_facts,test_facts = update_facts(train_facts,valid_facts,test_facts,ent_types,node_dict,rel_num,max_n)
    train_facts_tmp=train_facts
    valid_facts_tmp=valid_facts
    test_facts_tmp=test_facts
    all_facts = train_facts + valid_facts + test_facts
    train_facts = facts_to_id(train_facts, max_n, node_dict, rel_num, ent_num, ent_types, type_num)
    valid_facts = facts_to_id(valid_facts, max_n, node_dict, rel_num, ent_num, ent_types, type_num)
    test_facts = facts_to_id(test_facts, max_n, node_dict, rel_num, ent_num, ent_types, type_num)

    train_ground_truth,train_ground_truth_keys,train_max_type_num=get_ground_truth(train_facts_tmp, max_n, node_dict, ent_types, sparsifier, rel_num, ent_num, type_num)   
    train_ground_truth=helper(train_ground_truth)
    train_facts=update(train_facts,train_ground_truth,train_ground_truth_keys,train_max_type_num)

    valid_ground_truth,valid_ground_truth_keys,valid_max_type_num=get_ground_truth(valid_facts_tmp, max_n, node_dict, ent_types, sparsifier, rel_num, ent_num, type_num)   
    valid_ground_truth=helper(valid_ground_truth)
    valid_facts=update(valid_facts,valid_ground_truth,valid_ground_truth_keys,valid_max_type_num)  

    test_ground_truth,test_ground_truth_keys,test_max_type_num=get_ground_truth(test_facts_tmp, max_n, node_dict, ent_types, sparsifier, rel_num, ent_num, type_num)   
    test_ground_truth=helper(test_ground_truth)
    test_facts=update(test_facts,test_ground_truth,test_ground_truth_keys,test_max_type_num)  


    all_facts, all_fact_ids = get_truth_eval_new(all_facts, max_n, node_dict, ent_types, sparsifier, rel_num)

    input_info = dict()
    input_info['train_facts'] = train_facts
    input_info['valid_facts'] = valid_facts
    input_info['test_facts'] = test_facts
    input_info['ent_types'] = ent_types
    input_info['node_dict'] = node_dict
    input_info['node_num'] = node_num
    input_info['rel_num'] = rel_num
    input_info['ent_num'] = ent_num
    input_info['type_num'] = type_num
    input_info['max_n'] = max_n
    input_info['all_facts_eval'] = all_facts

    return input_info



def read_input(folder, sparsifier, entities_values2id):
    ins_info = get_input(folder + "/n-ary_train.json", folder + "/n-ary_valid.json", folder + "/n-ary_test.json",
                         folder + "/entity2types_ttv.txt", folder + "/type2relation2type_ttv.txt",
                         folder + "/type2relation2type2key2type_ttv.txt", sparsifier, entities_values2id)

    logger.info("Number of train facts: " + str(len(ins_info['train_facts'][0])))
    logger.info("Number of valid facts: " + str(len(ins_info['valid_facts'][0])))
    logger.info("Number of test facts: " + str(len(ins_info['test_facts'][0])))
    logger.info("Number of relations: " + str(ins_info['rel_num']))
    logger.info("Number of types: " + str(ins_info['type_num']))
    logger.info("Number of max_n: " + str(ins_info['max_n']))
    logger.info("Number of max_seq_length: " + str(2 * ins_info['max_n'] - 1))

    return ins_info

