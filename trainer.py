from models import MetaD
from dataloader import DataSet
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as DataLoader_graph
import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
import initial_feature

def index(y,pred,score):
    auc = roc_auc_score(y, score)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)
    return auc,precision,recall,f1

def evaluate(data_test,DrugLearner,device,model):
    test_pred = []
    test_scores=[] 
    test_y = []  
    
    for test_task in data_test:
        test_drug,test_support_x,test_support_y,test_query_x,test_query_y= [test_task[0]],[test_task[1][0]],test_task[1][1],[test_task[2][0]],test_task[2][1]

        if DrugLearner.lower() == 'mlp':
            test_drug = np.array([list(initial_feature.mol_feature_MACCSkey(i)) for i in test_drug])
            test_drug = torch.tensor(test_drug, dtype=torch.float, requires_grad=False,device=device)
        elif DrugLearner.lower() == 'gcn':
            data_list = [initial_feature.mol_feature_forgraph(i,device=device) for i in test_drug]
            test_drug = list(DataLoader_graph(data_list, batch_size=len(data_list), shuffle=False, ))[0]
        test_support_x = np.array([[initial_feature.target_feature_fingerprint(j) for j in i] for i in test_support_x])
        test_query_x = np.array([[initial_feature.target_feature_fingerprint(j) for j in i] for i in test_query_x])

        test_support_x = torch.tensor(test_support_x, dtype=torch.float, requires_grad=False,device=device)
        test_support_y = torch.tensor(test_support_y, dtype=torch.long, requires_grad=False,device=device)
        test_query_x = torch.tensor(test_query_x, dtype=torch.float, requires_grad=False,device=device)
        test_query_y = torch.tensor(test_query_y, dtype=torch.long, requires_grad=False,device=device)

        test_outcome = model(test_drug, test_support_x, test_support_y, test_query_x)
        test_y_pred = nn.functional.softmax(test_outcome.data,dim = 1).argmax(dim=1)
        test_pred_scores = nn.functional.softmax(test_outcome.data,dim = 1)[:,1]

        test_scores += test_pred_scores.cpu().numpy().tolist()
        test_pred += test_y_pred.cpu().numpy().tolist()
        test_y += test_query_y.cpu().numpy().tolist()

    test_auc, test_precision, test_recall, test_f1 = index(test_y,test_pred,test_scores)
    return test_auc, test_precision, test_recall, test_f1

def trainer(data = 'GPCR-one',device = "cpu",size_m = 200000,batch_size= 1024,
                lr = 0.001,inner_lr = 5,DrugMeta = True,DrugLearner = 'MLP'):
    
    DataTrain = DataSet(data = data, mode ='train', batchsz_m = size_m, k_shot = 1, k_query = 3)
    train_iter = DataLoader(dataset=DataTrain, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    DataValid = DataSet(data = data, mode = 'valid',k_shot = 1)
    data_valid = DataValid.dict_drugs_task

    device = torch.device(device)
    model = MetaD(drugmeta=DrugMeta,inner_lr=inner_lr,druglearner=DrugLearner,)
    model.to(device)
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = 0.0)

    train_epoch = 0  
    train_epoch_l =[] 
    train_loss_l = [] 
    for tasks in train_iter:
        drug, support_x, support_y, query_x, query_y = tasks[0], tasks[1], tasks[2], tasks[3], tasks[4]
        if DrugLearner.lower() == 'mlp':
            drug = np.array([list(initial_feature.mol_feature_MACCSkey(i)) for i in drug])  # drug: n*167 n:任务数
            drug = torch.tensor(drug, dtype=torch.float, requires_grad=False, device=device)
        elif DrugLearner.lower() == 'gcn':
            data_list = [initial_feature.mol_feature_forgraph(i,device=device) for i in drug]
            drug = list(DataLoader_graph(data_list, batch_size=len(data_list), shuffle=False))[0]

        support_x = np.array([[initial_feature.target_feature_fingerprint(j) for j in i] for i in support_x])  # (2*k_shot)*n*343
        query_x = np.array([[initial_feature.target_feature_fingerprint(j) for j in i] for i in query_x])  # (2*k_query)*n*343
        support_x = torch.tensor(support_x, dtype=torch.float, requires_grad=False,device=device).transpose(0, 1)
        support_y = support_y.reshape(-1).to(device)
        query_x = torch.tensor(query_x, dtype=torch.float, requires_grad=False,device=device).transpose(0, 1)
        query_y = query_y.reshape(-1).to(device)

        model.train()
        outcome = model(drug, support_x, support_y, query_x)
        loss = lossfunc(outcome, query_y,)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_l.append(loss.item())
        train_epoch += 1
        train_epoch_l.append(train_epoch)
        print(' Epoch:{}'.format(train_epoch),'\n','Loss:{}'.format(loss.item()))

        model.eval()

        DataValid = DataSet(data = data, mode = 'valid',k_shot = 1)
        data_valid = DataValid.dict_drugs_task
        auc, precision, recall, f1 = evaluate(data_valid,DrugLearner,device,model)
        print(' Valid ','AUC:',round(auc,5),' ','Precision:',round(precision,5),' ','Recall:',round(recall,5),' ','F1:',round(f1,5))

        DataTest = DataSet(data = data, mode = 'test',k_shot = 1)
        data_test = DataTest.dict_drugs_task
        auc, precision, recall, f1 = evaluate(data_test,DrugLearner,device,model)
        print(' Test ','AUC:',round(auc,5),' ','Precision:',round(precision,5),' ','Recall:',round(recall,5),' ','F1:',round(f1,5))

    return 0

