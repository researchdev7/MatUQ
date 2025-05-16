from __future__ import annotations
import csv
import random
import functools
import os
from typing import Sequence, Callable, Dict, Union, Optional, List
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure
from ase import Atoms
import schnetpack as spk
import schnetpack.properties as structure
from schnetpack.data import *
from schnetpack.model import AtomisticModel
import schnetpack.nn as snn
import schnetpack.transform as trn



def get_schnet_train_val_test_loader(dataset, train_indexs=None,
                              val_indexs=None, test_indexs=None,
                              collate_fn=default_collate,
                              batch_size=256, return_test=True,
                              num_workers=0, pin_memory=False):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_indexs: list
    val_indexs: list
    test_indexs: list
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    (test_loader): torch.utils.data.DataLoader
      returns if return_test=True.
    """
    total_size = len(dataset)
    train_indice = []
    val_indice = []
    test_indice = []
    for i in range(total_size):
        _, _cif_id = dataset[i]
        if _cif_id in train_indexs:
            train_indice.append(i)
        elif _cif_id in val_indexs:
            val_indice.append(i)
        elif _cif_id in test_indexs:
            test_indice.append(i)
        else:
            print("Can't find data which cif_id is %d in dataset", _cif_id)

    train_sampler = SubsetRandomSampler(train_indice)
    val_sampler = SubsetRandomSampler(val_indice)
    if return_test:
        test_sampler = SubsetRandomSampler(test_indice)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def atoms_collate_fn(dataset_list):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem, cif_id = dataset_list[0]
    idx_keys = {structure.idx_i, structure.idx_j, structure.idx_i_triples}
    # Atom triple indices must be treated separately
    idx_triple_keys = {structure.idx_j_triples, structure.idx_k_triples}

    batch_cif_ids = []
    batch = []
    for i, (elm, cif_id) in enumerate(dataset_list):
        batch.append(elm)
        batch_cif_ids.append(cif_id)

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[structure.n_atoms], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch[structure.n_atoms], dim=0
    )
    coll_batch[structure.idx_m] = idx_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[structure.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch, coll_batch["_targets"], batch_cif_ids

class LoadSchnetData(Dataset):
    def __init__(
        self,
        root_dir: str,
        task: str,
        random_seed: int = 123,
        format:Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        num_workers: int = 2,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.task = task
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, self.task, 'targets.csv')
        assert os.path.exists(id_prop_file), 'targets.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, self.task,
                                                   cif_id + '.cif'))
        if crystal is None:
            raise ValueError(f"Could not load crystal structure from crystal_file")

        atm = Atoms(
                numbers=crystal.atomic_numbers,
                positions=crystal.cart_coords,
                cell=crystal.lattice.matrix,
                pbc=True,
            )

        properties = {}
        _cif_id = int(cif_id)
        Z = atm.numbers.copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]], dtype=torch.long)
        properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
        properties[structure.position] = torch.tensor(atm.positions.copy(), dtype=torch.float32)
        properties[structure.cell] = torch.tensor(atm.cell[None].copy(), dtype=torch.float32)
        properties[structure.pbc] = torch.tensor(atm.pbc, dtype=torch.bool)
        properties["_targets"] = torch.Tensor([float(target)])
        properties = trn.ASENeighborList(cutoff=8.)(properties)
        return properties, _cif_id

class SchNet_Output(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "avg",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
        evidential="False",
        classification=False,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super(SchNet_Output, self).__init__()
        self.classification = classification
        self.evidential = evidential
        self.output_key = output_key
        if self.evidential == "True":
            self.model_outputs = ["mu","v","alpha","beta"]
        else:
            self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        self.act = nn.Softplus()
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        if self.classification:
            n_out = 2
        elif self.evidential=="True":
            n_out = 4
        else:
            n_out = 1

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode
        self.dropout = nn.Dropout(p=0.1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.cdropout = nn.Dropout()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        min_val = 1e-6
        # predict atomwise contributions
        y = self.outnet[:-1](inputs["scalar_representation"])
        y = self.dropout(y)
        y = self.outnet[-1](y)

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[structure.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[structure.n_atoms].unsqueeze(1)

        if self.classification:
            y = self.logsoftmax(y)
        if self.evidential=="True":
            if y.shape[0] == 4:
                y = torch.unsqueeze(y, 0)
            y = y.view(y.shape[0], -1, 4)
            mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(y, 1, dim=-1)]
            inputs["mu"] = mu
            inputs["v"] = self.act(logv) + min_val
            inputs["alpha"] = self.act(logalpha) + min_val+ 1
            inputs["beta"] = self.act(logbeta)+ min_val
        else:
            y = torch.squeeze(y)
            inputs[self.output_key] = y
        return inputs

class NeuralNetworkPotential(nn.Module):
    """
    A generic neural network potential class that sequentially applies a list of input
    modules, a representation module and a list of output modules.

    This can be flexibly configured for various, e.g. property prediction or potential
    energy sufaces with response properties.
    """

    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        evidential="False",
    ):
        """
        Args:
            representation: The module that builds representation from inputs.
            input_modules: Modules that are applied before representation, e.g. to
                modify input or add additional tensors for response properties.
            output_modules: Modules that predict output properties from the
                representation.
            postprocessors: Post-processing transforms that may be initialized using the
                `datamodule`, but are not applied during training.
            input_dtype_str: The dtype of real inputs.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__()
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules)
        self.output_modules = nn.ModuleList(output_modules)
        self.evidential = evidential

    def forward(self, inputs: Dict[str, torch.Tensor]):

        for m in self.input_modules:
            inputs = m(inputs)

        inputs = self.representation(inputs)

        for m in self.output_modules:
            inputs = m(inputs)
        if self.evidential=="True":
            return inputs["mu"], inputs["v"], inputs["alpha"], inputs["beta"]
        else:
            return inputs["y"]