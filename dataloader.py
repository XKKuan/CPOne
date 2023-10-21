import os
import random
from collections import Counter
import torch

class DataSet():

    def __init__(self, data = 'GPCR-one', mode = 'train', batchsz_m = 100000, k_shot = 1, k_query = 3):
        self.n_way = 2
        self.k_shot = k_shot
        self.k_query = k_query
        self.path = os.path.join('./DataSet',data, data + '_' + mode + '.txt')
        self.dict_drugs = self.loadTXT(self.path)
        self.dict_drugs_filter = self.data_filter(self.dict_drugs)
        self.batchsz_m = batchsz_m
        if mode == 'train':
            self.dict_drugs_task = self.create_TrainTask(self.dict_drugs_filter, self.batchsz_m)
        elif mode == 'test' or mode == 'valid':
            self.dict_drugs_task = self.create_TestTask(self.dict_drugs_filter)
        else:
            print('Please choose the appropriate mode!')

    def loadTXT (self, file):
        dict_drugs = {}
        with open(file,'r') as f:
            d = list(map(lambda x: x.strip().split(' '),f.readlines()))
            for item in d:
                dict_drugs[item[0]] = []
            for item in d:
                dict_drugs[item[0]].append(item[1:])
        return dict_drugs
    
    def data_filter(self,dict_drugs):
        dict_drugs_filter = {}
        margin = self.k_shot + self.k_query
        for i in dict_drugs.keys():
            l = []
            for j in dict_drugs[i]:
                l.append(j[-1])
            l_counter = Counter(l)
            if l_counter['0'] >=margin and l_counter['1']>=margin:
                dict_drugs_filter[i] = dict_drugs[i]
        return dict_drugs_filter
    
    def create_TrainTask(self,dict_drugs_filter,batchsz_m):
        dict_drugs_task = [] 
        drug_index = 0
        keys = list(dict_drugs_filter.keys())
        for i in range(batchsz_m):
            if drug_index == 0:
                random.shuffle(keys)
            current_drug_key = keys[drug_index]
            current_drug_value = dict_drugs_filter[current_drug_key]
            set_0 = [i[0] for i in current_drug_value if i[-1] =='0']
            set_1 = [i[0] for i in current_drug_value if i[-1] =='1']
            if len(set_0) < self.k_shot + self.k_query or len(set_1) < self.k_shot + self.k_query:
                drug_index = (drug_index + 1) % len(keys)
                continue
            set_0 = random.sample(set_0,self.k_shot + self.k_query)
            set_1 = random.sample(set_1,self.k_shot + self.k_query)
            support_0, query_0 = set_0[:self.k_shot], set_0[self.k_shot:]
            support_1, query_1 = set_1[:self.k_shot], set_1[self.k_shot:]
            support = [support_0, support_1]
            query = [query_0, query_1]
            dict_drugs_task.append([current_drug_key,[support, query]])
            drug_index = (drug_index+1) % len(keys)
        return dict_drugs_task
    
    def create_TestTask(self, dict_drugs_filter, ):
        dict_drugs_task = []
        keys = list(dict_drugs_filter.keys())
        for i in range(len(keys)):
            current_drug_key = keys[i]
            current_drug_value = dict_drugs_filter[current_drug_key]
            set_0 = [i[0] for i in current_drug_value if i[-1] == '0']
            set_1 = [i[0] for i in current_drug_value if i[-1] == '1']
            if len(set_0) < self.k_shot + self.k_query or len(set_1) < self.k_shot + self.k_query:
                continue
            support_0, query_0 = set_0[:self.k_shot], set_0[self.k_shot:]
            support_1, query_1 = set_1[:self.k_shot], set_1[self.k_shot:]
            support = support_0 + support_1
            support_y = [0] * len(support_0)+[1] * len(support_1)
            query = query_0 + query_1
            query_y = [0] * len(query_0)+[1] * len(query_1)
            dict_drugs_task.append([current_drug_key, [support,support_y],[query,query_y]])
        return dict_drugs_task
    
    def __getitem__(self, index):
        task = self.dict_drugs_task[index]
        drug = task[0]
        support_x = [ j for i in task[1][0] for j in i]
        support_y = torch.LongTensor([0] * len(task[1][0][0]) + [1] * len(task[1][0][1]))
        query_x   = [ j for i in task[1][1] for j in i]
        query_y   = torch.LongTensor([0] * len(task[1][1][0]) + [1] * len(task[1][1][1]))
        return (drug,support_x,support_y,query_x,query_y)
    
    def __len__(self):
        num = len(self.dict_drugs_task)
        return num
