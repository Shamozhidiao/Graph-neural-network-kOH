import os.path as osp
import numpy as np
import pandas as pd
from operator import itemgetter
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
# from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

import pdb


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


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    # AllChem.EmbedMolecule(mol)
    # AllChem.UFFOptimizeMolecule(mol)s
    N, M = mol.GetNumAtoms(), mol.GetNumBonds()

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())
    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

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
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MoleculeDataset(Dataset):
    def __init__(self, molecule_ds_path: str = None, smiles: List = None, log_k: List = None):
        # molecule_ds_path: *.csv
        super().__init__()
        if molecule_ds_path:
            molecule_df = pd.read_csv(molecule_ds_path)
            self.smiles = molecule_df['smiles'].tolist()
            # self.k = molecule_df['k'].tolist()
            self.log_k = molecule_df['log_k'].tolist()
        elif smiles and log_k:
            self.smiles = smiles
            self.log_k = log_k
        else:
            raise ValueError('please provide right arguments!')

    # def __len__(self):
    #     return len(self.smiles)

    # def __getitem__(self, idx):
    #     sample = self.smiles[idx]
    #     data = mol_to_graph(sample)
    #     return data, self.k[idx], self.log_k[idx]

    def len(self):
        return len(self.smiles)

    def get(self, idx):
        sample = self.smiles[idx]
        data = mol_to_graph(sample)
        # return data, self.k[idx], self.log_k[idx]
        return data, self.log_k[idx]


def random_split(smiles_list: List, train_ratio: float, val_ratio: float):
    dataset_len = len(smiles_list)
    train_cutoff = int(np.round(train_ratio * dataset_len))
    val_cutoff = int(np.round(val_ratio * dataset_len)) + train_cutoff

    indices = list(range(dataset_len))
    np.random.shuffle(indices)
    train_idx = sorted(indices[:train_cutoff])
    val_idx = sorted(indices[train_cutoff: val_cutoff])
    test_idx = sorted(indices[val_cutoff:])

    return train_idx, val_idx, test_idx


def scaffold_split(smiles_list: List, train_ratio: float, val_ratio: float):
    def generate_scaffold(smiles, include_chirality=False):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
        return scaffold

    dataset_len = len(smiles_list)
    train_cutoff = train_ratio * dataset_len
    val_cutoff = (train_ratio + val_ratio) * dataset_len

    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    train_idx, val_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(scaffold_set) > val_cutoff:
                test_idx.extend(scaffold_set)
            else:
                val_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    # pdb.set_trace()

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def stratified_split(y_list: List, train_ratio: float, val_ratio: float, random=False):
    y = np.array(y_list)
    sortidx = np.argsort(y)

    split_cd = 10
    train_cutoff = int(np.round(train_ratio * split_cd))
    val_cutoff = int(np.round(val_ratio * split_cd)) + train_cutoff

    train_idx = np.array([], dtype=int)
    val_idx = np.array([], dtype=int)
    test_idx = np.array([], dtype=int)

    while sortidx.shape[0] >= split_cd:
        sortidx_split, sortidx = np.split(sortidx, [split_cd])
        if random:
            shuffled = np.random.permutation(range(split_cd))
        else:
            shuffled = np.array(range(split_cd))
        train_idx = np.hstack(
            [train_idx, sortidx_split[shuffled[:train_cutoff]]])
        val_idx = np.hstack(
            [val_idx, sortidx_split[shuffled[train_cutoff: val_cutoff]]])
        test_idx = np.hstack(
            [test_idx, sortidx_split[shuffled[val_cutoff:]]])

    if sortidx.shape[0] > 0:
        train_idx = np.hstack([train_idx, sortidx])

    # pdb.set_trace()

    train_idx = np.sort(train_idx).tolist()
    val_idx = np.sort(val_idx).tolist()
    test_idx = np.sort(test_idx).tolist()

    return train_idx, val_idx, test_idx


class MoleculeDatasetWrapper(object):
    def __init__(self, dataset_path, batch_size, num_workers, record_path='./', split='random', normalize_k=False):
        super().__init__()
        self.molecule_ds_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.record_path = record_path
        self.split = split
        self.normalize_k = normalize_k

    def get_train_val_test_data_loaders(self):
        molecule_df = pd.read_csv(self.molecule_ds_path)
        smiles_list = molecule_df['smiles'].tolist()
        log_k_list = molecule_df['log_k'].tolist()

        train_r, val_r = 0.8, 0.1  # fixed ratio: 0.8, 0.1, 0.1 (train, val, test)

        if self.split == 'random':
            train_idx, val_idx, test_idx = \
                random_split(smiles_list=smiles_list, train_ratio=train_r, val_ratio=val_r)
        elif self.split == 'scaffold':
            train_idx, val_idx, test_idx = \
                scaffold_split(smiles_list=smiles_list, train_ratio=train_r, val_ratio=val_r)
        elif self.split == 'stratified':
            train_idx, val_idx, test_idx = \
                stratified_split(y_list=log_k_list, train_ratio=train_r, val_ratio=val_r)
        elif self.split == 'random_stratified':
            train_idx, val_idx, test_idx = \
                stratified_split(y_list=log_k_list, train_ratio=train_r, val_ratio=val_r, random=True)
        else:
            raise NotImplementedError('please provide right split method!')

        # pdb.set_trace()

        if self.normalize_k:
            mean_value = molecule_df['log_k'].mean()
            std_value = molecule_df['log_k'].std()
            print(mean_value, std_value)
            molecule_df['log_k'] = (molecule_df['log_k'] - mean_value) / std_value

        train_mol_df = molecule_df.loc[train_idx]
        val_mol_df = molecule_df.loc[val_idx]
        test_mol_df = molecule_df.loc[test_idx]

        train_mol_ds_path = osp.join(self.record_path, 'train_molecule_dataset.csv')
        val_mol_ds_path = osp.join(self.record_path, 'val_molecule_dataset.csv')
        test_mol_ds_path = osp.join(self.record_path, 'test_molecule_dataset.csv')

        train_mol_df.to_csv(train_mol_ds_path, index=False)
        val_mol_df.to_csv(val_mol_ds_path, index=False)
        test_mol_df.to_csv(test_mol_ds_path, index=False)

        # train_dataset = MoleculeDataset(smiles=[smiles_list[i] for i in train_idx], log_k=[log_k[i] for i in train_idx])
        # val_dataset = MoleculeDataset(smiles=[smiles_list[i] for i in val_idx], log_k=[log_k[i] for i in val_idx])
        # test_dataset = MoleculeDataset(smiles=[smiles_list[i] for i in test_idx], log_k=[log_k[i] for i in test_idx])
        train_dataset = MoleculeDataset(molecule_ds_path=train_mol_ds_path)
        val_dataset = MoleculeDataset(molecule_ds_path=val_mol_ds_path)
        test_dataset = MoleculeDataset(molecule_ds_path=test_mol_ds_path)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, drop_last=False, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, drop_last=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, drop_last=False, pin_memory=True)

        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    mol_ds_path = './new_molecule_dataset.csv'

    # dataset = MoleculeDataset(mol_ds_path)
    # for i in range(10):
    #     data, log_k = dataset[i]
    #     print(f'sample {i} - x shape: {data.x.shape}, edge index shape: {data.edge_index.shape}, edge attr shape: {data.edge_attr.shape}, log_k: {log_k:.4f}')

    dataset = MoleculeDatasetWrapper(mol_ds_path, batch_size=32, num_workers=4, split='stratified')
    train_l, val_l, test_l = dataset.get_train_val_test_data_loaders()

    def count_loader(loader):
        count = 0
        for _, log_k in loader:
            count += log_k.size(0)
        return count

    print(count_loader(train_l), count_loader(val_l), count_loader(test_l))

    for data, log_k in train_l:
        print(data.x.shape, data.edge_index.shape, data.edge_attr.shape)
        print(log_k.shape)
        break
