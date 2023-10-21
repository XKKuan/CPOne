import numpy as np
from rdkit.Chem import MACCSkeys
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from pybiomed_helper import CalculateConjointTriad
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def mol_feature_forgraph(mol,device = 'cpu'):
    mol = Chem.MolFromSmiles(mol)
    mol = Chem.AddHs(mol)  
    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()
    type_idx = []
    chirality_idx = []  
    atomic_number = []
    ATOM_LIST = list(range(1, 119))  
    CHIRALITY_LIST = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]  
    BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]  
    BONDDIR_LIST = [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))  
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())
    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1).to(device)  

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long,device=device) 
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long,device=device) 
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,)
    return data

def mol_feature_MACCSkey(mol):
    x = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(mol))
    return x

def target_feature_fingerprint(target, type_name= 'Conjoint_Triad_fp'):
    features = []
    try:
        if type_name == "Conjoint_Triad_fp":
            features = CalculateConjointTriad(target)
    except:
        print("Fingerprint Conjoint_Triad not working for this sequence")
    return np.array(features)