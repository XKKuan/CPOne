import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GCN(nn.Module):
    def __init__(self,num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0.2, pool='mean'):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        num_atom_type = 119
        num_chirality_tag = 3
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GCNConv(emb_dim,emb_dim))
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        return h

class DrugLearner(nn.Module):
    def __init__(self,model = 'MLP', out_dim = 32, drop_ratio = 0.2,gcn_numlayer = 5,gcn_pool = 'mean',gcn_emb_dim = 300):
        super(DrugLearner, self).__init__()
        if model.lower() == 'mlp':
            self.druglearner = nn.Sequential(
                OrderedDict([('fc1',nn.Linear(167,128)),
                             ('relu',nn.ReLU()),
                             ('dropout', nn.Dropout(drop_ratio)),
                             ('fc3',nn.Linear(128,64)),
                             ('relu',nn.ReLU()),
                             ('dropout', nn.Dropout(drop_ratio)),
                             ('fc2',nn.Linear(64,out_dim))])
            )
            nn.init.xavier_normal_(self.druglearner.fc1.weight)
            nn.init.xavier_normal_(self.druglearner.fc2.weight)
            nn.init.xavier_normal_(self.druglearner.fc3.weight)
        elif model.lower() == 'gcn':
            self.druglearner = GCN(num_layer=gcn_numlayer, emb_dim=gcn_emb_dim,feat_dim=out_dim, drop_ratio=drop_ratio,pool=gcn_pool)
        else:
            raise ValueError('No this Model!')
    def forward(self,drug):
        drug = self.druglearner(drug)
        return drug 

class TargetLearner(nn.Module):  
    def __init__(self,out_dim = 32, drop_ratio = 0.2):
        super(TargetLearner, self).__init__()
        self.atomlearner = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(343, 64)),
                         ('relu', nn.ReLU()),
                         ('dropout',nn.Dropout(drop_ratio)),
                         ('fc2', nn.Linear(64, out_dim))])
        )
        nn.init.xavier_normal_(self.atomlearner.fc1.weight)
        nn.init.xavier_normal_(self.atomlearner.fc2.weight)
    def forward(self,atom):
        atom = self.atomlearner(atom)
        return atom 

class MetaD(nn.Module):
                                     
    def __init__(self,in_dim_drug = 128, in_dim_target = 32,drop_ratio = 0.2,druglearner = 'MLP',drugmeta = True,inner_lr = 4,):
        super(MetaD, self).__init__()

        self.drugmeta = drugmeta
        self.druglearner = DrugLearner(model = druglearner,out_dim = in_dim_drug, drop_ratio = drop_ratio,gcn_numlayer = 5,gcn_pool = 'mean',gcn_emb_dim = 300)
        self.targetlearner = TargetLearner(out_dim = in_dim_target, drop_ratio = drop_ratio)
        self.inner_lr =inner_lr
        self.inner_loss_func = nn.CrossEntropyLoss()
        self.cf = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(in_dim_target+in_dim_drug, 32)),
                         ('relu', nn.ReLU()),
                         ('dropout', nn.Dropout(drop_ratio)),
                         ('fc2', nn.Linear(32, 2))])
        )
        nn.init.xavier_normal_(self.cf.fc1.weight)
        nn.init.xavier_normal_(self.cf.fc2.weight)

    def forward(self,drug,support_x,support_y,query_x):
        drug = self.druglearner(drug)
        support_atom = self.targetlearner(support_x)
        drug.retain_grad()
        drug_s = drug.reshape(drug.shape[0], 1, drug.shape[1])
        drug_s = drug_s.expand(-1, support_atom.shape[1],-1)
        comb_s = torch.cat([drug_s,support_atom],dim=-1)
        pred = self.cf(comb_s)
        loss_inner = self.inner_loss_func(pred.reshape(-1,2),support_y)
        loss_inner.backward(retain_graph=True)
        self.zero_grad()
        grad_meta = drug.grad
        if self.drugmeta:
            drug_q = drug - self.inner_lr * grad_meta
        else:
            drug_q = drug
        drug_q = drug_q.reshape(drug_q.shape[0],1,drug_q.shape[1])
        query_atom = self.targetlearner(query_x)
        drug_q = drug_q.expand(-1, query_atom.shape[1], -1)
        comb_q = torch.cat([drug_q, query_atom], dim=-1)
        pred_q = self.cf(comb_q)

        return pred_q.reshape(-1,2)
