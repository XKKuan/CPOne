import argparse
from rdkit import RDLogger
import trainer
import torch
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='kinase-one', nargs="?",help="Which dataset to use GPCR-one/kinase-one")
parser.add_argument("--device",type =str,default="cuda:0",nargs="?",help="cpu,cuda:0,cuda:1...")
parser.add_argument("--size_m", type=int, default=200000, nargs="?",help="the number of mission.")
parser.add_argument("--batch_size", type=int, default=1024, nargs="?",help="batch size")
parser.add_argument("--lr", type=float, default=0.001, nargs="?",help="learning rate ")
parser.add_argument("--meta_rate", type=float, default=5.0, nargs="?",help="meta rate ")
parser.add_argument("--DrugMeta",type=bool,default=True,nargs='?',help="use DrugMeta or not.")
parser.add_argument("--DrugLearner",type=str,default='GCN',nargs='?',help="Which drug learner to use MLP/GCN.")
args = parser.parse_args()
RDLogger.DisableLog('rdApp.*')
   
SEED = 3
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
random.seed(SEED)

trainer.trainer(data = args.data,device = args.device,size_m = args.size_m,batch_size= args.batch_size,
                lr = args.lr,inner_lr = args.meta_rate,DrugMeta = args.DrugMeta,DrugLearner = args.DrugLearner)

