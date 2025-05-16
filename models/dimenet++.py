import os
import os.path as osp
from functools import partial
from math import pi as PI
from math import sqrt
import csv
import functools
import random
import json
from typing import Callable, Dict, Optional, Tuple, Union
from pymatgen.core.structure import Structure
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import Embedding, Linear
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.utils import scatter

def get_dimenetpp_train_val_test_loader(dataset, train_indexs=None,
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
        _, _, cif_id = dataset[i]
        _cif_id = int(cif_id)
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


def collate_dimenetpp(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.

      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    Z: torch.LongTensor shape (N, )
      Atom type
    position: torch.Tensor shape (N, 3)
      Atomic position
    batch: torch.LongTensor shape (N, )
      Batch indices assigning each atom
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_z, batch_pos =  [], []
    batch, batch_target = [], []
    batch_cif_ids = []
    for i, ((Z, position), target, cif_id)\
            in enumerate(dataset_list):
        N = Z.shape[0]  # number of atoms for this crystal
        batch_z.append(Z)
        batch_pos.append(position)
        batch.append(torch.tensor([i]*N, dtype=torch.long))
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
    return (torch.cat(batch_z, dim=0),
            torch.cat(batch_pos, dim=0),
            torch.cat(batch, dim=0)),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class LoadDimeNetPPData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    task: str
        The task of the dataset
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, task, random_seed=123):
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
        Z = torch.tensor(crystal.atomic_numbers, dtype=torch.long)
        position = torch.tensor(crystal.cart_coords, dtype=torch.float32)
        target = torch.Tensor([float(target)])
        return (Z, position), target, cif_id


class Envelope(torch.nn.Module):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: Tensor) -> Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 +
                c * x_pow_p2) * (x < 1.0).to(x.dtype)


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial: int, cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        import sympy as sym

        from torch_geometric.nn.models.dimenet_utils import (
            bessel_basis,
            real_sph_harm,
        )

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(partial(self._sph_to_tensor, sph1))
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    @staticmethod
    def _sph_to_tensor(sph, x: Tensor) -> Tensor:
        return torch.zeros_like(x) + sph

    def forward(self, dist: Tensor, angle: Tensor, idx_kj: Tensor) -> Tensor:
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = Linear(num_spherical * num_radial, num_bilinear,
                              bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(
            torch.empty(hidden_channels, num_bilinear, hidden_channels))

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor) -> Tensor:
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor) -> Tensor:
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(torch.nn.Module):
    def __init__(
            self,
            num_radial: int,
            hidden_channels: int,
            num_layers: int,
            act: Callable,
            output_initializer: str = 'zeros',
            evidential: str = 'False',
            classification: bool = False,
            dropout: bool = False,
    ):
        assert output_initializer in {'zeros', 'glorot_orthogonal'}

        super().__init__()

        self.classification = classification
        self.evidential = evidential
        self.dropout = dropout

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        if self.classification:
            self.lin = Linear(hidden_channels, 1, bias=False)
            self.softmax = nn.Sigmoid()
            # self.softmax = nn.LogSoftmax(dim=1)
        elif self.evidential == "True":
            self.lin = Linear(hidden_channels, 4, bias=False)
        else:
            self.lin = Linear(hidden_channels, 1, bias=False)
        self.drop = nn.Dropout(p=0.1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_initializer == 'zeros':
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == 'glorot_orthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        for lin in self.lins:
            x = self.act(lin(x))
        if self.dropout:
            x = self.drop(x)
        return self.lin(x)


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = 'zeros',
        evidential: str = 'False',
        classification: bool = False,
        dropout: bool = False,
    ):
        assert output_initializer in {'zeros', 'glorot_orthogonal'}

        super().__init__()

        self.classification = classification
        self.evidential = evidential
        self.dropout = dropout
        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))

        if self.classification:
            self.lin = Linear(out_emb_channels, 1, bias=False)
            self.softmax = nn.Sigmoid()
            # self.softmax = nn.LogSoftmax(dim=1)
        elif self.evidential == "True":
            self.lin = Linear(out_emb_channels, 4, bias=False)
        else:
            self.lin = Linear(out_emb_channels, 1, bias=False)
        self.drop = nn.Dropout(p=0.1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_initializer == 'zeros':
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == 'glorot_orthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        if self.dropout:
            x = self.drop(x)
        return self.lin(x)


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class DimeNet(torch.nn.Module):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.
    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    .. note::

        For an example of using a pretrained DimeNet variant, see
        `examples/qm9_pretrained_dimenet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_dimenet.py>`_.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
            (default: :obj:`"zeros"`)
    """
    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        output_initializer: str = 'zeros',
        evidential: str = 'False',
        classification: bool = False,
    ):
        super().__init__()

        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        act = activation_resolver(act)

        self.classification = classification
        self.evidential = evidential
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(
                num_radial,
                hidden_channels,
                num_output_layers,
                act,
                output_initializer,
                evidential,
                classification,
            ) for _ in range(num_blocks)]+[OutputBlock(
                num_radial,
                hidden_channels,
                num_output_layers,
                act,
                output_initializer,
                evidential,
                classification,
                dropout=True,
            )])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(
                hidden_channels,
                num_bilinear,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])
        self.out_act = nn.Softplus()
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        min_val = 1e-6
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        if isinstance(self, DimeNetPlusPlus):
            pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
            a = (pos_ij * pos_jk).sum(dim=-1)
            b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        elif isinstance(self, DimeNet):
            pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        out = scatter(P, batch, dim=0, reduce='sum')

        if self.evidential == "True":
            if out.shape[0] == 4:
                out = torch.unsqueeze(out, 0)
            out = out.view(out.shape[0], -1, 4)
            mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(out, 1, dim=-1)]
            return mu, self.out_act(logv)+ min_val, self.out_act(logalpha)+ min_val + 1, self.out_act(logbeta)+ min_val
        else:
            return torch.squeeze(out)


class DimeNetPlusPlus(DimeNet):
    r"""The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    :class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model with
    8x faster and 10% more accurate than :class:`DimeNet`.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation funtion.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
            (default: :obj:`"zeros"`)
    """

    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        output_initializer: str = 'zeros',
        evidential: str = 'False',
        classification: bool = False,

    ):
        act = activation_resolver(act)

        super().__init__(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            num_bilinear=1,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
            evidential=evidential,
            classification=classification
        )

        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initalization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:
        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                num_output_layers,
                act,
                output_initializer,
                evidential,
                classification,
            ) for _ in range(num_blocks)]+[OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                num_output_layers,
                act,
                output_initializer,
                evidential,
                classification,
                dropout=True,
            )])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])

        self.reset_parameters()