import argparse
from rdkit import RDLogger
import trainer
import torch
import random
import numpy as np
   
SEED = 3
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
random.seed(SEED)

trainer.trainer(data = 'kinase-one',device = 'cuda:0',size_m = 200000,batch_size= 1024,
                lr = 0.001,inner_lr = 4.0,DrugMeta = True,DrugLearner = 'GCN')

