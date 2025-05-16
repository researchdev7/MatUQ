import sys
import os
import csv
import math
import copy
import warnings
from typing import TypeVar, Type, List, Optional, Tuple, Union, Callable
import numpy as np
from operator import truediv
import pymatgen
from pymatgen.core.structure import Structure
import torch
from torch import Tensor, cos_
import torch.nn as nn
from torch.nn import Parameter, Module, Linear, Dropout, LayerNorm, ModuleList, Identity
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_, normal_
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from .cuda_funcs import *

def generate_site_species_vector(structure: pymatgen.core.structure.Structure, ATOM_NUM_UPPER):
    if hasattr(structure, 'species'):
        atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        atom_num = torch.tensor(structure.atomic_numbers, dtype=torch.long).unsqueeze_(-1)
        x_species_vector = torch.eye(ATOM_NUM_UPPER)[atom_num - 1].squeeze()

    else:
        x_species_vector = []
        for site in structure.species_and_occu:
            site_species_and_occupancy = []
            # サイトの各元素について、one-hot encodingした上で占有率をかけて、元素ごとの占有率ベクトルを計算
            for elem in site.elements:
                if type(elem) == pymatgen.core.Element:
                    occupancy = site.element_composition[elem]
                elif type(elem) == pymatgen.core.periodic_table.Specie or type(
                        elem) == pymatgen.core.periodic_table.Species:
                    occupancy = site.element_composition[elem.element]
                elif type(elem) == pymatgen.core.composition.Composition:
                    occupancy = site.element_composition[elem.element]
                    # print(elem, occupancy)
                elif type(elem) == pymatgen.core.periodic_table.DummySpecie or type(
                        elem) == pymatgen.core.periodic_table.DummySpecies:
                    raise ValueError(f'Unsupported specie: {site}! Skipped')
                else:
                    print(site, type(site))
                    raise AttributeError
                atom_num = torch.tensor(elem.Z, dtype=torch.long)
                elem_onehot = torch.eye(ATOM_NUM_UPPER)[atom_num - 1]
                site_species_and_occupancy.append(elem_onehot * occupancy)
            # サイトの各元素についてのone-hot vectorのsumとって、サイトごとの占有率に変換
            site_species_and_occupancy_sum = torch.stack(site_species_and_occupancy).sum(0)
            x_species_vector.append(site_species_and_occupancy_sum)
        x_species_vector = torch.stack(x_species_vector, 0)

    if x_species_vector.dim() == 1:
        x_species_vector.unsqueeze_(0)
    return x_species_vector

def get_crystalframer_train_val_test_loader(dataset, train_indexs=None,
                              val_indexs=None, test_indexs=None,
                              batch_size=256, return_test=True,
                              num_workers=0, pin_memory=False):

    total_size = len(dataset)
    train_indice = []
    val_indice = []
    test_indice = []
    for i in range(total_size):
        cif_id = dataset[i].structure_id
        _cif_id = int(cif_id)
        if _cif_id in train_indexs:
            train_indice.append(i)
        elif _cif_id in val_indexs:
            val_indice.append(i)
        elif _cif_id in test_indexs:
            test_indice.append(i)
        else:
            print("Can't find data which cif_id is %d in dataset", _cif_id)

    train_dataset = Subset(dataset, train_indice)
    val_dataset = Subset(dataset, val_indice)
    test_dataset = Subset(dataset, test_indice)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


##Fetch dataset; processes the raw data if specified
def CrystalFramerData(data_path, task):

    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    if os.path.exists(os.path.join(data_path, task, "crystalframer_data.pt")) == True:
        dataset = StructureDataset(
            data_path,
            task,
        )
    else:
        process_data(data_path, task)
        dataset = StructureDataset(
            data_path,
            task,
        )
    return dataset

##Dataset class from pytorch/pytorch geometric; inmemory case
class StructureDataset(InMemoryDataset):
    def __init__(
            self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ["crystalframer_data.pt"]
        return file_names

def process_data(data_path, task):
    ##Begin processing data
    print("Processing data to: " + os.path.join(data_path, task))
    assert os.path.exists(data_path), "Data path not found in " + data_path

    ##Load targets
    target_property_file = os.path.join(data_path, task, "targets.csv")
    assert os.path.exists(target_property_file), (
            "targets not found in " + target_property_file
    )
    with open(target_property_file) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader]

    ##Process structure files and create structure graphs
    data_list = []
    for index in range(0, len(target_data)):

        structure_id = target_data[index][0]
        crystal = Structure.from_file(os.path.join(
            data_path, task, structure_id + ".cif"))

        atom_pos = torch.tensor(crystal.cart_coords, dtype=torch.float)
        atom_fea = generate_site_species_vector(crystal, 98)
        target = target_data[index][1]
        y = torch.Tensor([float(target)])
        data = Data(x=atom_fea, y=y, pos=atom_pos)
        data.trans_vec = torch.tensor(crystal.lattice.matrix, dtype=torch.float)[None]
        data.structure_id = structure_id
        data.sizes = torch.tensor([atom_pos.shape[0]], dtype=torch.long)
        data_list.append(data)

    if os.path.isdir(os.path.join(data_path, task)) == False:
        os.mkdir(os.path.join(data_path, task))

    ##Save processed dataset to file
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(data_path, task, "crystalframer_data.pt"))

REPRODUCIBLITY_STATE = 2

def max_pool(x, batch, sizes):
    x = torch.split_with_sizes(x, sizes.tolist(), 0)
    x = torch.stack([torch.max(x,dim=0)[0] for x in x])
    return x

def avr_pool(x, batch, sizes):
    if REPRODUCIBLITY_STATE>=1 and CUPY_AVAILABLE:
        x = IrregularMeanCUDA.apply(x, batch, sizes)
    else:
        x = torch.split_with_sizes(x, sizes.tolist(), 0)
        x = torch.stack([torch.mean(x,dim=0) for x in x])
    return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

Entity = TypeVar('Entity', bound='LatticeformerParams')

class LatticeformerParams:
    def __init__(self,
                 domain: str = "real",
                 lattice_range: int = 4,
                 minimum_range: bool = True,
                 adaptive_cutoff_sigma: float = -3.5,
                 gauss_lb_real: float = 0.5,
                 gauss_lb_reci: float = 0.5,
                 scale_real: List[float] = [1.4],
                 scale_reci: List[float] = [2.2],
                 normalize_gauss: bool = True,
                 value_pe_dist_real: int = 64,
                 value_pe_dist_coef: float = 1.0,
                 value_pe_dist_max: float = -10.0,
                 value_pe_dist_wscale: float = 1.0,
                 value_pe_wave_real: int = 0,
                 value_pe_dist_reci: int = 0,
                 value_pe_wave_reci: int = 0,
                 value_pe_angle_real: int = 16,
                 value_pe_angle_coef: float = 1.0,
                 value_pe_angle_wscale: float = 4.0,
                 positive_func_beta: float = 0.1,
                 layer_index: int = -1,
                 gauss_state: str = "q",
                 frame_method: str = "max",
                 frame_mode: str = "both",
                 cos_abs: int = 1,
                 symm_break_noise: float = 1e-5,
                 ) -> None:

        self.layer_index = layer_index
        self.domain = domain
        self.cos_abs = cos_abs
        self.lattice_range = lattice_range
        self.minimum_range = minimum_range
        self.adaptive_cutoff_sigma = adaptive_cutoff_sigma
        self.gauss_lb_real = gauss_lb_real
        self.gauss_lb_reci = gauss_lb_reci
        self.scale_real = scale_real
        self.scale_reci = scale_reci
        self.normalize_gauss = normalize_gauss
        self.value_pe_dist_real = value_pe_dist_real
        self.value_pe_dist_coef = value_pe_dist_coef
        self.value_pe_dist_max = value_pe_dist_max
        self.value_pe_dist_wscale = value_pe_dist_wscale
        self.value_pe_wave_real = value_pe_wave_real
        self.value_pe_dist_reci = value_pe_dist_reci
        self.value_pe_wave_reci = value_pe_wave_reci
        self.value_pe_angle_real = value_pe_angle_real
        self.value_pe_angle_coef = value_pe_angle_coef
        self.value_pe_angle_wscale = value_pe_angle_wscale
        self.positive_func_beta = positive_func_beta
        self.gauss_state = gauss_state
        self.frame_mode = frame_mode
        self.frame_method = frame_method
        self.symm_break_noise = symm_break_noise

    def parseFromArgs(self, args):
        for key in self.__dict__:
            self.__dict__[key] = getattr(args, key, self.__dict__[key])
        print("Parsed LatticeformerParams:")
        print(self.__dict__)

    def getLayerParameters(self, layer_index) -> Entity:
        if self.domain in ("real", "reci", "multihead"):
            domain = self.domain
        else:
            domains = self.domain.split('-')
            domain = domains[layer_index % len(domains)]

        scale_real = self.scale_real
        scale_reci = self.scale_reci
        if isinstance(scale_real, (list, tuple)):
            scale_real = scale_real[layer_index % len(scale_real)]
        if isinstance(scale_reci, (list, tuple)):
            scale_reci = scale_reci[layer_index % len(scale_reci)]

        params = copy.deepcopy(self)
        params.domain = domain
        params.scale_real = scale_real
        params.scale_reci = scale_reci
        params.layer_index = layer_index
        return params

def _in_projection(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        b_q: Optional[Tensor] = None,
        b_k: Optional[Tensor] = None,
        b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    Dq, Dk, Dv = w_q.size(0), w_k.size(0), w_v.size(0)
    assert w_q.shape[1] == Eq, f"expecting query weights shape of (*, {Eq}), but got {w_q.shape}"
    assert w_k.shape[1] == Ek, f"expecting key weights shape of (*, {Ek}), but got {w_k.shape}"
    assert w_v.shape[1] == Ev, f"expecting value weights shape of (*, {Ev}), but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Dq,), f"expecting query bias shape of {(Dq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Dk,), f"expecting key bias shape of {(Dk,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Dv,), f"expecting value bias shape of {(Dv,)}, but got {b_v.shape}"

    # F.linear(x, W, b) = xW^T + b
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        batch: Tensor,
        batch_kv: Tensor,
        edges: Tensor,
        attn_weights: Optional[Tensor] = None,
        values: Tensor = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        edges: index pairs (i,j) to define attentions between q and p,v.
        attn_weights: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(Nt, B, E)` where Nt is the target sequence length, B is batch size,
            and E is embedding dimension.
        - key: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - value: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - edges: :math:`(2, M)` where M is the edge num.
        - attn_weights: `(M, B)` where M in the edge num, B is batch size.
        - Output: attention values have shape :math:`(Nt, B, E)`; attention weights
            have shape :math:`(M, B)` where M in the edge num, B is batch size.
    """
    Nt, B, E = q.shape
    q = q / math.sqrt(E)
    # (M, B, E) x (M, B, E) -> (M, B)
    if REPRODUCIBLITY_STATE >= 2 and dropout_p == 0.0 and batch is batch_kv:
        output = FusedDotProductAttentionCUDA.apply(
            q, k, v, attn_weights, values, batch, edges,
        )
        return output, None, None

    # Perform dot product attention using nested_tensor code.
    # Deprecated: much slower than a naive implementation using split and loop.
    attn = (q[edges[0]] * k[edges[1]]).sum(dim=-1)

    # flag = torch.are_deterministic_algorithms_enabled()
    # torch.use_deterministic_algorithms(False)
    bsz = batch.max().item() + 1
    q_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
    q_sizes.scatter_add_(0, batch, torch.ones_like(batch))

    if batch_kv is batch:
        k_sizes = q_sizes
    else:
        k_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
        k_sizes.scatter_add_(0, batch_kv, torch.ones_like(batch_kv))
    # This is because self-attention has the same number of queries and keys (sys_size).
    edg_sizes = q_sizes * k_sizes

    q_sizes = q_sizes.tolist()
    k_sizes = k_sizes.tolist()
    edg_sizes = edg_sizes.tolist()
    # torch.use_deterministic_algorithms(flag)

    if True:
        # The scaled_dot operation involves the summations along the key axis
        # whose size varies among batch samples. So we split concatenated data
        # into a list of batch samples and apply the scaled_dot for each sample.
        # We could do the same without the splitting & looping by using scatter_add,
        # but we rather avoid scatter_add as it breaks reproducibility in backprop.
        if attn_weights is None:
            attn_weights = 0

        # standard normalization for all system points.
        attn += attn_weights
        attn = torch.split_with_sizes(attn, edg_sizes)
        attn = torch.cat(
            [F.softmax(a.view(qs, ks, -1), dim=1).view(qs * ks, -1) for a, qs, ks in zip(attn, q_sizes, k_sizes)])

        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)

        # (M, B, 1) x (M, B, E) -> (q_len, B, E)
        if values is None:
            output = attn[..., None] * v[edges[1]]
        else:
            output = attn[..., None] * (v[edges[1]] + values)
        output = torch.split_with_sizes(output, edg_sizes)
        output = torch.cat([o.view((qs, ks) + o.shape[1:]).sum(dim=1) for o, qs, ks in zip(output, q_sizes, k_sizes)])

    sizes = (q_sizes, k_sizes, edg_sizes)

    return output, attn, sizes


def _scaled_dot_product_attention_sigmamat(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        batch: Tensor,
        batch_kv: Tensor,
        edges: Tensor,
        attn_weights: Optional[Tensor] = None,
        values: Tensor = None,
        dropout_p: float = 0.0
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        edges: index pairs (i,j) to define attentions between q and p,v.
        attn_weights: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(Nt, B, E)` where Nt is the target sequence length, B is batch size,
            and E is embedding dimension.
        - key: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - value: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - edges: :math:`(2, M)` where M is the edge num.
        - attn_weights: `(M, B)` where M in the edge num, B is batch size.
        - Output: attention values have shape :math:`(Nt, B, E)`; attention weights
            have shape :math:`(M, B)` where M in the edge num, B is batch size.
    """
    Nt, B, E = q.shape
    q = q / math.sqrt(E)
    # (M, B, E) x (M, B, E) -> (M, B)
    if REPRODUCIBLITY_STATE >= 2 and dropout_p == 0.0 and batch is batch_kv:
        output = FusedDotProductAttentionSigmaMatCUDA.apply(
            q, k, v, attn_weights, values, batch, edges,
        )
        return output, None, None

    # Perform dot product attention using nested_tensor code.
    # Deprecated: much slower than a naive implementation using split and loop.
    attn = (q[edges[0]] * k[edges[1]]).sum(dim=-1)

    # flag = torch.are_deterministic_algorithms_enabled()
    # torch.use_deterministic_algorithms(False)
    bsz = batch.max().item() + 1
    q_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
    q_sizes.scatter_add_(0, batch, torch.ones_like(batch))

    if batch_kv is batch:
        k_sizes = q_sizes
    else:
        k_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
        k_sizes.scatter_add_(0, batch_kv, torch.ones_like(batch_kv))
    # This is because self-attention has the same number of queries and keys (sys_size).
    edg_sizes = q_sizes * k_sizes

    q_sizes = q_sizes.tolist()
    k_sizes = k_sizes.tolist()
    edg_sizes = edg_sizes.tolist()
    # torch.use_deterministic_algorithms(flag)

    if True:
        # The scaled_dot operation involves the summations along the key axis
        # whose size varies among batch samples. So we split concatenated data
        # into a list of batch samples and apply the scaled_dot for each sample.
        # We could do the same without the splitting & looping by using scatter_add,
        # but we rather avoid scatter_add as it breaks reproducibility in backprop.
        if attn_weights is None:
            attn_weights = 0

        # standard normalization for all system points.
        attn += attn_weights
        attn = torch.split_with_sizes(attn, edg_sizes)
        attn = torch.cat(
            [F.softmax(a.view(qs, ks, -1), dim=1).view(qs * ks, -1) for a, qs, ks in zip(attn, q_sizes, k_sizes)])

        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)

        # (M, B, 1) x (M, B, E) -> (q_len, B, E)
        if values is None:
            output = attn[..., None] * v[edges[1]]
        else:
            output = attn[..., None] * (v[edges[1]] + values)
        output = torch.split_with_sizes(output, edg_sizes)
        output = torch.cat([o.view((qs, ks) + o.shape[1:]).sum(dim=1) for o, qs, ks in zip(output, q_sizes, k_sizes)])

    sizes = (q_sizes, k_sizes, edg_sizes)

    return output, attn, sizes

def compute_real_domain_attn_low_memory(alpha, pos, dist2_min, tvecs, batch, edges, proj, K: int, dist_max: float,
                                        wscale: float, frame_method: str, rvecs: Tensor = None, cutoff: float = None):
    # alpha: (nodes, heads)
    # dist2: (edge_num, R)
    # pe_wave: (edge_num, R, K)
    # where R = (1+2r)^3 and K is the PE dim.
    adap_cutoff_args = ()
    if cutoff is not None:
        rvlen = torch.norm(rvecs, 2, dim=-1)
        adap_cutoff_args = (rvlen, cutoff)

    proj_ = proj
    # Give proj_=None to disable internal v'=Wv projection.
    # proj_ = None
    out = RealPeriodicEncodingWithProjFuncCUDA.apply(
        alpha, pos, dist2_min, tvecs, batch, edges, K, dist_max, wscale, proj_, frame_method,
        *adap_cutoff_args
    )
    attn_weights = out[0]
    values = out[1] if len(out) >= 2 else None
    sigma_mat = out[2] if len(out) >= 3 else None
    sigma_mat_max = out[3] if len(out) >= 4 else None

    # Do the projection inside the kernel to reduce memory usage
    if values is not None and (proj is not None and proj_ is None):
        # values: (edges, heads, pe_dim)
        # proj  : (edges, heads, head_dim, pe_dim)
        # values = (values[:, :, None, :]*proj).sum(axis=-1)
        values = proj @ values[..., None]
        values.squeeze_(-1)

    return attn_weights, values, sigma_mat, sigma_mat_max


def compute_frame_maximum_method_(q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, proj, K: int, dist_max: float,
                                  wscale: float, frame_method: str, this_epoch: int = None, rvecs: Tensor = None,
                                  cutoff: float = None):
    # alpha: (nodes, heads)
    # dist2: (edge_num, R)
    # pe_wave: (edge_num, R, K)
    # where R = (1+2r)^3 and K is the PE dim.
    adap_cutoff_args = ()
    if cutoff is not None:
        rvlen = torch.norm(rvecs, 2, dim=-1)
        adap_cutoff_args = (rvlen, cutoff)

    proj_ = proj
    Nt, B, E = q.shape
    q = q / math.sqrt(E)

    N, H = alpha.shape

    # Give proj_=None to disable internal v'=Wv projection.
    # proj_ = None

    if frame_method == "max" or frame_method == "max_wo_orthogonal":
        frame1 = ComputeFrame1MaximumMethodCUDA.apply(
            q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, K, dist_max, wscale, *adap_cutoff_args
        )

        frame2_max = ComputeFrame2MaximumMethodCUDA.apply(
            q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, K, dist_max, wscale, frame1, *adap_cutoff_args
        )
    elif frame_method == "max_static":
        frame1 = ComputeFrame1MaximumStaticMethodCUDA.apply(
            q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, K, dist_max, wscale, *adap_cutoff_args
        )

        frame2_max = ComputeFrame2MaximumStaticMethodCUDA.apply(
            q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, K, dist_max, wscale, frame1,
            *adap_cutoff_args
        )
    else:
        raise NotImplementedError()

    if frame_method == "max" or frame_method == "max_static":
        dot_product = frame1.reshape(N, H, -1, 3) @ frame2_max.reshape(N, H, 3, -1)
        dot_product = dot_product[:, :, :, 0].repeat(1, 1, 3)

        frame2 = torch.nn.functional.normalize(frame2_max - dot_product * frame1, dim=2)

        frame3 = torch.linalg.cross(frame1, frame2)

    elif frame_method == "max_wo_orthogonal":
        frame2 = frame2_max
        frame3 = ComputeFrame3MaximumMethodCUDA.apply(
            q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, K, dist_max, wscale, frame1, frame2, *adap_cutoff_args
        )

    else:
        raise NotImplementedError()

    return frame1, frame2, frame3


def compute_frame_lattice_method_(q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, proj, K: int, dist_max: float,
                                  wscale: float, frame_method: str, this_epoch: int = None, rvecs: Tensor = None,
                                  cutoff: float = None):
    # alpha: (nodes, heads)
    # dist2: (edge_num, R)
    # pe_wave: (edge_num, R, K)
    # where R = (1+2r)^3 and K is the PE dim.
    adap_cutoff_args = ()
    if cutoff is not None:
        rvlen = torch.norm(rvecs, 2, dim=-1)
        adap_cutoff_args = (rvlen, cutoff)

    proj_ = proj
    Nt, B, E = q.shape
    q = q / math.sqrt(E)

    # Give proj_=None to disable internal v'=Wv projection.
    # proj_ = None

    frame1, frame2, frame3 = ComputeLatticeFrameCUDA.apply(
        q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, K, dist_max, wscale, *adap_cutoff_args
    )

    return frame1, frame2, frame3


def compute_frame_pca_method_(q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, proj, K: int, dist_max: float,
                              wscale: float, frame_method: str, this_epoch: int = None, rvecs: Tensor = None,
                              cutoff: float = None):
    adap_cutoff_args = ()
    if cutoff is not None:
        rvlen = torch.norm(rvecs, 2, dim=-1)
        adap_cutoff_args = (rvlen, cutoff)

    proj_ = proj
    Nt, B, E = q.shape
    q = q / math.sqrt(E)

    frame1, frame2, frame3 = ComputePCAFrame.apply(
        q, k, v, alpha, pos, dist2_min, tvecs, batch, edges, K, dist_max, wscale, *adap_cutoff_args
    )

    return frame1, frame2, frame3


def compute_real_domain_framed_proj(alpha, pos, dist2_min, tvecs, batch, edges, proj, proj_angle1, proj_angle2,
                                    proj_angle3, frame_vec_1, frame_vec_2, frame_vec_3, K: int, dist_max: float,
                                    wscale: float, dim_angle_enc: int, value_pe_angle_scale: float, cos_abs: int,
                                    length_rbf_mul: float, angle_rbf_mul: float, rvecs: Tensor = None,
                                    cutoff: float = None):
    adap_cutoff_args = ()
    if cutoff is not None:
        rvlen = torch.norm(rvecs, 2, dim=-1)
        adap_cutoff_args = (rvlen, cutoff)
    proj_ = proj

    out = RealEncodingWithFramedProjFuncCUDA.apply(
        alpha, pos, dist2_min, tvecs, batch, edges, K, dist_max, wscale, proj_, proj_angle1, proj_angle2, proj_angle3,
        frame_vec_1, frame_vec_2, frame_vec_3,
        dim_angle_enc, value_pe_angle_scale, cos_abs, length_rbf_mul, angle_rbf_mul, *adap_cutoff_args
    )

    return out


def compute_reci_domain_attn(alpha, kr_base, rvecs, vcell, batch, edges):
    # We compute: sum k'[exp(-k^2/4a^2) * cos(kr)] for the real-domain func exp(-a^2*r^2).
    # Here, 1/a^2 = alpha in our code. So,
    #   sum k'[exp(-alpha^2*k^2/4) * cos(kr)]
    # cos_kr: (E, R)
    # k2    : (N, R)
    # alpha : (L, H)
    # vcell : (N)

    attn_weights = ReciPeriodicEncodingFuncCUDA.apply(
        alpha,
        kr_base,
        rvecs,
        vcell,
        batch,
        edges,
    )
    return attn_weights, None
    # self_coef = vcell.unsqueeze(1)[batch] / (2.0*math.pi*alpha)**(3/2)   # (L, H)
    alp = alpha.unsqueeze(2)  # (L, H, 1)
    k2 = k2.unsqueeze(1)[batch]  # (L, 1, R)
    cos_kr = cos_kr.unsqueeze(1)  # (E, 1, R)

    alpha_k2 = alp * k2
    alpha_k2_max = alpha_k2.max()
    attn_weights = (torch.exp(alpha_k2 - alpha_k2_max)[edges0] * cos_kr).sum(dim=-1)
    attn_weights = torch.log(attn_weights.clamp_(1e-6)) + alpha_k2_max
    # NOTE: using below instead of above yeilds nan
    # attn_weights_ = (torch.exp((alpha*k2)[edges0])*cos_kr).sum(dim=-1)
    # attn_weights_ = torch.log(attn_weights_.clamp_(1e-6))

    # NOTE: the following correction is required when using non-softmax attention.
    # In the paper, the correction term is a factor: (1/V)(2*pi/gamma)^(3/2).
    # Here, alpha = -1/(2*gamma). So, (2*pi/gamma)^3/2 = (-4*pi*alpha)^3/2.
    log_ci = (3 / 2) * (torch.log((-4 * math.pi) * alpha)) - torch.log(vcell)[batch, None]
    attn_weights += log_ci[edges0]

    return attn_weights, None


def compute_real_domain_attn_frame_proj(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        batch_q: Tensor,
        batch_kv: Tensor,
        edges: Tensor,
        pos: Tensor,
        dist2_min: Tensor,
        tvecs: Tensor,
        rvecs: Tensor,
        pe_dist_proj: Optional[Tensor],
        proj_angle1: Optional[Tensor],
        proj_angle2: Optional[Tensor],
        proj_angle3: Optional[Tensor],
        alpha: float,
        params: LatticeformerParams = None,
        this_epoch: int = None
):
    K = params.value_pe_dist_real
    dist_max = params.value_pe_dist_max
    if dist_max < 0:
        dist_max = (-dist_max) * params.scale_real
    wscale = params.value_pe_dist_wscale
    adap_args = ()
    if params.adaptive_cutoff_sigma < 0:
        # negative values to adaptive cutoff using actual alpha values.
        adap_args = (rvecs, params.adaptive_cutoff_sigma)
    elif params.adaptive_cutoff_sigma > 0:
        adap_args = (rvecs, params.scale_real * params.gauss_lb_real ** (-0.5) * params.adaptive_cutoff_sigma)

    if params.frame_method in ["weighted_pca", "no"]:
        attn_weights, values, sigma_mat, sigma_mat_max_tmp = compute_real_domain_attn_low_memory(
            alpha, pos, dist2_min, tvecs, batch_q, edges, pe_dist_proj,
            K, dist_max, wscale, params.frame_method, *adap_args
        )

    elif params.frame_method in ["max", "max_wo_orthogonal", "max_static"]:
        v_dummy = torch.zeros((v.shape[0], v.shape[1], 3), device=q.device)
        frame_vec_1, frame_vec_2, frame_vec_3 = compute_frame_maximum_method_(
            q, k, v_dummy, alpha, pos, dist2_min, tvecs, batch_q,
            edges, pe_dist_proj, K, dist_max, wscale, params.frame_method, this_epoch, *adap_args
        )
        del v_dummy
        sigma_mat = None

    elif params.frame_method == "lattice":
        v_dummy = torch.zeros((v.shape[0], v.shape[1], 3), device=q.device)
        frame_vec_1, frame_vec_2, frame_vec_3 = compute_frame_lattice_method_(
            q, k, v_dummy, alpha, pos, dist2_min, tvecs, batch_q,
            edges, pe_dist_proj, K, dist_max, wscale, params.frame_method, this_epoch, *adap_args
        )
        del v_dummy
        sigma_mat = None

    elif params.frame_method == "pca":
        v_dummy = torch.zeros((v.shape[0], v.shape[1], 3), device=q.device)
        frame_vec_1, frame_vec_2, frame_vec_3 = compute_frame_pca_method_(
            q, k, v_dummy, alpha, pos, dist2_min, tvecs, batch_q,
            edges, pe_dist_proj, K, dist_max, wscale, params.frame_method, this_epoch, *adap_args
        )
        del v_dummy
        sigma_mat = None

    else:
        raise NotImplementedError()

    if sigma_mat is not None:
        v_dummy = torch.zeros((v.shape[0], v.shape[1], 6), device=q.device)
        sigma_mat_fused_tmp = _scaled_dot_product_attention_sigmamat(
            q, k, v_dummy, batch_q, batch_kv, edges, attn_weights, sigma_mat, 0.0)[0]
        sigma_mat_fused = torch.zeros((sigma_mat_fused_tmp.shape[0], sigma_mat_fused_tmp.shape[1], 3, 3),
                                      device=sigma_mat_fused_tmp.device)

        sigma_mat_fused[:, :, 0, 0] = sigma_mat_fused_tmp[:, :, 0]
        sigma_mat_fused[:, :, 0, 1] = sigma_mat_fused_tmp[:, :, 1]
        sigma_mat_fused[:, :, 0, 2] = sigma_mat_fused_tmp[:, :, 2]
        sigma_mat_fused[:, :, 1, 0] = sigma_mat_fused_tmp[:, :, 1]
        sigma_mat_fused[:, :, 1, 1] = sigma_mat_fused_tmp[:, :, 3]
        sigma_mat_fused[:, :, 1, 2] = sigma_mat_fused_tmp[:, :, 4]
        sigma_mat_fused[:, :, 2, 0] = sigma_mat_fused_tmp[:, :, 2]
        sigma_mat_fused[:, :, 2, 1] = sigma_mat_fused_tmp[:, :, 4]
        sigma_mat_fused[:, :, 2, 2] = sigma_mat_fused_tmp[:, :, 5]

        determinant = torch.linalg.det(sigma_mat_fused.detach())
        if params.symm_break_noise > 0:
            sigma_mat_max = torch.zeros((sigma_mat_fused_tmp.shape[0], sigma_mat_fused_tmp.shape[1], 3, 3),
                                        device=sigma_mat_fused_tmp.device)
            sigma_mat_max[:, :, 0, 0] = sigma_mat_max_tmp[:, :, 0]
            sigma_mat_max[:, :, 0, 1] = sigma_mat_max_tmp[:, :, 1]
            sigma_mat_max[:, :, 0, 2] = sigma_mat_max_tmp[:, :, 2]
            sigma_mat_max[:, :, 1, 0] = sigma_mat_max_tmp[:, :, 1]
            sigma_mat_max[:, :, 1, 1] = sigma_mat_max_tmp[:, :, 3]
            sigma_mat_max[:, :, 1, 2] = sigma_mat_max_tmp[:, :, 4]
            sigma_mat_max[:, :, 2, 0] = sigma_mat_max_tmp[:, :, 2]
            sigma_mat_max[:, :, 2, 1] = sigma_mat_max_tmp[:, :, 4]
            sigma_mat_max[:, :, 2, 2] = sigma_mat_max_tmp[:, :, 5]
            sigma_mat_fused += sigma_mat_max * determinant.pow(1 / 3).reshape(
                determinant.shape[0], determinant.shape[1], 1, 1) * params.symm_break_noise

        eig_vec = torch.eye(3, device=sigma_mat_fused.device).repeat(sigma_mat_fused.shape[0], sigma_mat_fused.shape[1],
                                                                     1, 1)
        bool_det = determinant > 1e-5
        if torch.sum(bool_det) != 0:
            eig_vec[bool_det, :, :] = torch.linalg.eigh(sigma_mat_fused.detach()[bool_det, :, :])[1]
        frame_vec = torch.real(eig_vec)
        frame_vec_1 = frame_vec[:, :, 0]
        frame_vec_2 = frame_vec[:, :, 1]
        frame_vec_3 = frame_vec[:, :, 2]

        dev = frame_vec.device
        N = frame_vec_1.shape[0]
        H = frame_vec_1.shape[1]

        random_variable = torch.tensor([-1, 1], device=dev)
        idx = torch.randint(2, size=(N, H, 1), device=dev)
        random_variable_first = random_variable[idx].repeat(1, 1, 3)
        idx = torch.randint(2, size=(N, H, 1), device=dev)
        random_variable_second = random_variable[idx].repeat(1, 1, 3)
        idx = torch.randint(2, size=(N, H, 1), device=dev)
        random_variable_third = random_variable[idx].repeat(1, 1, 3)

        frame_vec_1 = frame_vec_1 * random_variable_first
        frame_vec_2 = frame_vec_2 * random_variable_second
        frame_vec_3 = frame_vec_3 * random_variable_third

        frame_vec = torch.stack([frame_vec_1, frame_vec_2, frame_vec_3], dim=-1)
        detval = torch.linalg.det(frame_vec).unsqueeze(-1).unsqueeze(-1)
        frame_vec = frame_vec * detval

        frame_vec_1 = frame_vec[:, :, :, 0]
        frame_vec_2 = frame_vec[:, :, :, 1]
        frame_vec_3 = frame_vec[:, :, :, 2]

        frame_vec_1 = frame_vec_1.to(torch.float32).contiguous()
        frame_vec_2 = frame_vec_2.to(torch.float32).contiguous()
        frame_vec_3 = frame_vec_3.to(torch.float32).contiguous()

        del v_dummy

    if params.frame_method in ["weighted_pca", "max", "lattice", "pca", "max_wo_orthogonal", "max_static"]:
        attn_weights, values = compute_real_domain_framed_proj(
            alpha, pos, dist2_min, tvecs, batch_q, edges, pe_dist_proj, proj_angle1,
            proj_angle2, proj_angle3, frame_vec_1, frame_vec_2, frame_vec_3, K, dist_max, wscale,
            params.value_pe_angle_real, params.value_pe_angle_wscale,
            params.cos_abs, params.value_pe_dist_coef, params.value_pe_angle_coef, *adap_args)

    return attn_weights, values


def crystalformer_multi_head_attention_forward_cuda(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        batch_q: Tensor,
        batch_kv: Tensor,
        edges: Tensor,
        pos: Tensor,
        dist2_min: Tensor,
        tvecs: Tensor,
        kr_base: Tensor,
        rvecs: Tensor,
        vcell: Tensor,
        lattice_pos_weights: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        q_proj_weight: Tensor,
        k_proj_weight: Tensor,
        v_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        pe_dist_proj: Optional[Tensor],
        pe_wave_proj: Optional[Tensor],
        proj_angle1: Optional[Tensor],
        proj_angle2: Optional[Tensor],
        proj_angle3: Optional[Tensor],
        training: bool = True,
        need_weights: bool = True,
        gauss_scale: Optional[Tensor] = None,
        atten_scale: Optional[Tensor] = None,
        onehot: Optional[Tensor] = None,
        params: LatticeformerParams = None,
        this_epoch: int = None
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        dist: distance matrices of points of lattices.
        lattice_pos_weights: weights for lattice position embeddings.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - dist: :math:`(N, L, S, R)` or `(L, S, R)`, where N is the batch size, S is the source sequence length,
           N is the batch size, R is the number of neighbors of the lattice.
        - lattice_pos_weights: :math:`(E)`, where E is the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    # _mha_shape_check(query, key, value, edges, dist2, cos_kr, k2, vcell, selfe, num_heads)

    # set up shape vars
    tgt_len, embed_dim = query.shape
    src_len, _ = key.shape
    esz = edges.shape[1]
    # assert embed_dim == embed_dim_to_check, \
    #     f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    # if isinstance(embed_dim, torch.Tensor):
    #     # embed_dim can be a tensor when JIT tracing
    #     head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    # else:
    #     head_dim = embed_dim // num_heads
    # assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    head_dim_q = q_proj_weight.shape[0] // num_heads
    head_dim_k = k_proj_weight.shape[0] // num_heads
    head_dim_v = v_proj_weight.shape[0] // num_heads
    assert head_dim_q * num_heads == q_proj_weight.shape[0]
    assert head_dim_k * num_heads == k_proj_weight.shape[0]
    assert head_dim_v * num_heads == v_proj_weight.shape[0]

    # allow MHA to have different embedding dimensions when separate projection weights are used
    assert key.shape[0] == value.shape[0], \
        f"key's sequence and batch dims {key.shape[0]} do not match value's {value.shape[0]}"

    #
    # compute in-projection
    #

    assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
    assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
    assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
    if in_proj_bias is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = in_proj_bias.split([q_proj_weight.size(0), k_proj_weight.size(0), v_proj_weight.size(0)])
    q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    #
    # reshape q, k, v for multihead attention
    #
    q = q.contiguous().view(tgt_len, num_heads, head_dim_q)
    k = k.contiguous().view(k.shape[0], num_heads, head_dim_k)
    v = v.contiguous().view(v.shape[0], num_heads, head_dim_v)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # compute attn_weights according to dist
    # TODO: this sites is the largest bottleneck of latticeformer.
    # dist: (esz, neighbors)
    # q: (tgt_len, num_heads, head_dim)
    # lattice_pos_weights: (num_heads*head_dim)
    if params is None or params.domain == "no":
        attn_weights = None
        values = None
    else:
        domain = params.domain
        gauss_lb_real = params.gauss_lb_real
        gauss_lb_reci = params.gauss_lb_reci
        beta = params.positive_func_beta

        # exp( -pos_func(alpha)*distance^2 )
        if params.gauss_state.startswith("q"):
            alpha = q.reshape(tgt_len, num_heads, 1, head_dim_q) @ lattice_pos_weights.reshape(num_heads, head_dim_q, 1)
        elif params.gauss_state == "1":
            ones = torch.ones_like(q)
            alpha = ones.reshape(tgt_len, num_heads, 1, head_dim_q) @ lattice_pos_weights.reshape(num_heads, head_dim_q,
                                                                                                  1)
        else:
            # query is original state 'x' before q = Wx
            # lattice_pos_weights's shape is (num_heads, embed_dim)
            alpha = F.linear(query, lattice_pos_weights)

        alpha = alpha.view(tgt_len, num_heads)
        if params.normalize_gauss:
            scale_available = gauss_scale[0] > 0
            if scale_available:
                scale = gauss_scale[0]
                shift = gauss_scale[1]
            else:
                scale = torch.reciprocal(alpha.detach().std())
                shift = -alpha.detach().mean()
                # print("Computed gauss scale and shift:", scale, shift)
            alpha += shift
            alpha *= scale

            # save the scale only once in the first training step.
            if training and not scale_available:
                torch.nn.init.constant_(gauss_scale[0], scale)
                torch.nn.init.constant_(gauss_scale[1], shift)
                print("Saved gauss scale and shift:", gauss_scale.data)

        func = lambda x, lb: (1.0 - lb) * F.elu(x * (beta / (1.0 - lb))) + 1.0

        # a = alpha
        # c = func(alpha, gauss_lb_real)
        # print("layer\t{:.4f}\t{:.4f}\t{:.4f} | {:.4f}\t{:.4f}\t{:.4f}" .format(a.mean(), a.median(), a.std(), c.mean(), c.median(), c.std()))

        '''
        Saved gauss scale and shift: tensor([ 2.5387e+01, -6.5168e-03], device='cuda:0')
        Saved gauss scale and shift: tensor([3.1654e+01, 5.2495e-04], device='cuda:0')
        Saved gauss scale and shift: tensor([3.0025e+01, 7.2772e-03], device='cuda:0')
        Saved gauss scale and shift: tensor([2.2007e+01, 3.1712e-03], device='cuda:0')
        '''

        if domain == "real":
            alpha = func(alpha, gauss_lb_real) * (-0.5 * params.scale_real ** -2)
            attn_weights, values = compute_real_domain_attn_frame_proj(
                q, k, v, batch_q, batch_kv, edges, pos, dist2_min,
                tvecs, rvecs, pe_dist_proj, proj_angle1, proj_angle2, proj_angle3,
                alpha, params, this_epoch
            )
        elif domain == "reci":
            alpha = func(alpha, gauss_lb_reci) * (-0.5 * params.scale_reci ** 2)
            attn_weights, values = compute_reci_domain_attn(alpha, kr_base, rvecs, vcell, batch_q, edges)
        elif domain == "RECI":
            # NOTE: for func_test, use this version.
            alpha = 1 / func(alpha, gauss_lb_reci) * (-0.5 * params.scale_reci ** 2)
            attn_weights, values = compute_reci_domain_attn(alpha, kr_base, rvecs, vcell, batch_q, edges)
        elif domain == "multihead":
            a1, a2 = alpha.chunk(2, 1)
            a1 = func(a1, gauss_lb_real) * (-0.5 * params.scale_real ** -2)
            a2 = func(a2, gauss_lb_reci) * (-0.5 * params.scale_reci ** 2)
            w1, v1 = compute_real_domain_attn_frame_proj(
                q, k, v, batch_q, batch_kv, edges, pos, dist2_min,
                tvecs, rvecs, pe_dist_proj[:, :num_heads // 2],
                proj_angle1, proj_angle2, proj_angle3,
                a1, params, this_epoch
            )
            w2, v2 = compute_reci_domain_attn(a2, kr_base, rvecs, vcell, batch_q, edges)
            attn_weights = torch.cat([w1, w2], dim=1)
            alpha = torch.cat([a1, a2], dim=1)
            if v1 is None and v2 is None:
                values = None
            elif v1 is None:
                values = torch.cat([torch.zeros_like(v2), v2], dim=1)
            elif v2 is None:
                values = torch.cat([v1, torch.zeros_like(v1)], dim=1)
            else:
                values = torch.cat([v1, v2], dim=1)
        else:
            raise NotImplementedError(f"Not implmeneted for domain = {domain}.")

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights, sizes = _scaled_dot_product_attention(q, k, v, batch_q, batch_kv, edges,
                                                                            attn_weights, values, dropout_p)

    attn_output = attn_output.contiguous().view(tgt_len, head_dim_v * num_heads)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(esz, num_heads)
        attn_output_weights = attn_output_weights.mean(dim=1)
        return attn_output, attn_output_weights
    else:
        return attn_output, values

class NonDynamicallyQuantizableLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        try:
            super().__init__(in_features, out_features, bias=bias,
                            device=device, dtype=dtype)
        except:
            super().__init__(in_features, out_features, bias=bias)

class CrystalformerMultiheadAttentionCUDA(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 kdim=None, vdim=None,
                 params=LatticeformerParams(),
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CrystalformerMultiheadAttentionCUDA, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim * num_heads if kdim is not None and kdim > 0 else embed_dim
        self.vdim = vdim * num_heads if vdim is not None and vdim > 0 else embed_dim
        assert params.domain in ("real", "reci", "RECI", "multihead", "no")
        if params.domain == "multihead":
            assert num_heads % 2 == 0, "In multihead mode, num_head must be even as MHA of each domain uses num_head/2 of heads."

        # lattice parameters that are referenced in indexed_lattice_multi_head_attention_forward.
        self.params = params
        self.gauss_scale = Parameter(torch.zeros(2, **factory_kwargs), requires_grad=False)
        self.atten_scale = Parameter(torch.zeros(1, **factory_kwargs), requires_grad=False)

        self.num_heads = num_heads
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((self.vdim, embed_dim), **factory_kwargs))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(self.kdim * 2 + self.vdim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(self.vdim, embed_dim, bias=bias, **factory_kwargs)

        if params.domain == "no":
            self.lattice_pos_weights = None
        elif params.gauss_state.startswith("q"):
            self.lattice_pos_weights = Parameter(torch.empty((self.kdim), **factory_kwargs))
        elif params.gauss_state == "1":
            self.lattice_pos_weights = Parameter(torch.empty((self.kdim), **factory_kwargs))
        elif params.gauss_state == "x-xn":
            self.lattice_pos_weights = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
            self.lattice_pos_weights2 = Parameter(torch.empty((self.kdim), **factory_kwargs))
        elif params.gauss_state.startswith("x"):
            self.lattice_pos_weights = Parameter(torch.empty((self.num_heads, embed_dim), **factory_kwargs))

        self.ATOM_NUM = 98
        head = self.num_heads
        cond = 1

        if params.domain in ["real", "multihead"]:
            self.pe_dist_proj = Parameter(
                torch.empty(cond, head, self.head_dim, params.value_pe_dist_real, **factory_kwargs)) \
                if params.value_pe_dist_real > 0 else None
            self.pe_wave_proj = Parameter(
                torch.empty(cond, head, self.head_dim, params.value_pe_wave_real ** 3, **factory_kwargs)) \
                if params.value_pe_wave_real > 0 else None
            if params.domain == "real":
                self.proj_angle1 = Parameter(
                    torch.empty(cond, head, self.head_dim, params.value_pe_angle_real, **factory_kwargs)) \
                    if params.value_pe_angle_real > 0 else None
                self.proj_angle2 = Parameter(
                    torch.empty(cond, head, self.head_dim, params.value_pe_angle_real, **factory_kwargs)) \
                    if params.value_pe_angle_real > 0 else None
                self.proj_angle3 = Parameter(
                    torch.empty(cond, head, self.head_dim, params.value_pe_angle_real, **factory_kwargs)) \
                    if params.value_pe_angle_real > 0 else None
            else:
                self.proj_angle1 = Parameter(
                    torch.empty(cond, head // 2, self.head_dim, params.value_pe_angle_real, **factory_kwargs)) \
                    if params.value_pe_angle_real > 0 else None
                self.proj_angle2 = Parameter(
                    torch.empty(cond, head // 2, self.head_dim, params.value_pe_angle_real, **factory_kwargs)) \
                    if params.value_pe_angle_real > 0 else None
                self.proj_angle3 = Parameter(
                    torch.empty(cond, head // 2, self.head_dim, params.value_pe_angle_real, **factory_kwargs)) \
                    if params.value_pe_angle_real > 0 else None

        elif params.domain in ["reci", "RECI"]:
            self.pe_dist_proj = Parameter(
                torch.empty(cond, head, self.head_dim, params.value_pe_dist_reci, **factory_kwargs)) \
                if params.value_pe_dist_reci > 0 else None
            self.pe_wave_proj = Parameter(
                torch.empty(cond, head, self.head_dim, params.value_pe_wave_reci ** 3, **factory_kwargs)) \
                if params.value_pe_wave_reci > 0 else None
            self.proj_angle1 = None
            self.proj_angle2 = None
            self.proj_angle3 = None

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.out_proj.weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

        # TODO: consider the normalization method later
        if self.lattice_pos_weights is not None:
            if self.params.gauss_state in ("q", "q-headdim", "1"):
                normal_(self.lattice_pos_weights, 0., (self.kdim // self.num_heads) ** -0.5)
            elif self.params.gauss_state == "q-kdim":
                normal_(self.lattice_pos_weights, 0., (self.kdim) ** -0.5)
            elif self.params.gauss_state in ("x", "x-norm"):
                normal_(self.lattice_pos_weights, 0., (self.embed_dim) ** -0.5)
            elif self.params.gauss_state == "x-xavier":
                xavier_uniform_(self.lattice_pos_weights)
            elif self.params.gauss_state == "x-xn":
                xavier_uniform_(self.lattice_pos_weights)
                normal_(self.lattice_pos_weights2, 0., (self.kdim // self.num_heads) ** -0.5)
            elif self.params.gauss_state == "x-xn2":
                H, K, D = self.num_heads, self.kdim, self.embed_dim
                W = self.lattice_pos_weights
                W1 = torch.empty((K, D), device=W.device, dtype=W.dtype)
                W2 = torch.empty((H, 1, K // H), device=W.device, dtype=W.dtype)
                xavier_uniform_(W1)
                normal_(W2, 0., (K // H) ** -0.5)
                W0 = W2 @ W1.reshape((H, K // H, D))
                with torch.no_grad():
                    self.lattice_pos_weights.set_(W0.reshape_as(W))
            else:
                raise NotImplementedError()

        # D0 = 1/sqrt(I + O)
        # D1 = 1/sqrt(I + O*H)
        # I = 64, O = 16, H = 8
        # D0 = 1/sqrt(64 + 16) = 1/( 4*sqrt(5) )
        # D1 = 1/sqrt(64 + 16*8) = 1/( 8*sqrt(3) )
        # D1/D0 = 0.645497224
        # good case = 0.0099
        # current (no t-fixup)  = 0.0063
        if self.pe_dist_proj is not None:
            with torch.no_grad():
                for i, W in enumerate(self.pe_dist_proj):
                    W = W.view(-1, W.shape[-1])
                    co, ci = W.shape
                    a = (self.embed_dim + ci) / (co + ci)
                    # 'a' is to keep the scale regardless of pe_headed True or False.
                    xavier_uniform_(W, (ci) ** -0.5)

        if self.proj_angle1 is not None:
            with torch.no_grad():
                for i, W in enumerate(self.proj_angle1):
                    W = W.view(-1, W.shape[-1])
                    co, ci = W.shape
                    a = (self.embed_dim + ci) / (co + ci)
                    # 'a' is to keep the scale regardless of pe_headed True or False.
                    xavier_uniform_(W, (ci) ** -0.5)
        if self.proj_angle2 is not None:
            with torch.no_grad():
                for i, W in enumerate(self.proj_angle2):
                    W = W.view(-1, W.shape[-1])
                    co, ci = W.shape
                    a = (self.embed_dim + ci) / (co + ci)
                    # 'a' is to keep the scale regardless of pe_headed True or False.
                    xavier_uniform_(W, (ci) ** -0.5)
        if self.proj_angle3 is not None:
            with torch.no_grad():
                for i, W in enumerate(self.proj_angle3):
                    W = W.view(-1, W.shape[-1])
                    co, ci = W.shape
                    a = (self.embed_dim + ci) / (co + ci)
                    # 'a' is to keep the scale regardless of pe_headed True or False.
                    xavier_uniform_(W, (ci) ** -0.5)

        if self.pe_wave_proj is not None:
            with torch.no_grad():
                for i, W in enumerate(self.pe_wave_proj):
                    W = W.view(-1, W.shape[-1])
                    co, ci = W.shape
                    a = (self.embed_dim + ci) / (co + ci)
                    # 'a' is to keep the scale regardless of pe_headed True or False.
                    xavier_uniform_(W, (ci) ** -0.5)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, batch_q: Tensor, batch_kv: Tensor, edges: Tensor,
                points: Tensor, dist2_min: Tensor, tvecs: Tensor,
                kr_base: Optional[Tensor] = None, rvecs: Optional[Tensor] = None, vcell: Optional[Tensor] = None,
                onehot: Tensor = None,
                need_weights: bool = True, this_epoch: int = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """

        W = self.lattice_pos_weights
        if self.params.gauss_state == "x-xn":
            H, K, D = self.num_heads, self.kdim, self.embed_dim
            W2 = self.lattice_pos_weights2.view(H, 1, K // H)
            W = W2 @ W.reshape((H, K // H, D))
            W = W.reshape(H, D)

        attn_output, attn_output_weights = crystalformer_multi_head_attention_forward_cuda(
            query, key, value, batch_q, batch_kv, edges,
            points, dist2_min, tvecs,
            kr_base, rvecs, vcell,
            W,
            self.embed_dim, self.num_heads,
            self.q_proj_weight, self.k_proj_weight, self.v_proj_weight,
            self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            self.pe_dist_proj,
            self.pe_wave_proj,
            self.proj_angle1,
            self.proj_angle2,
            self.proj_angle3,
            training=self.training,
            need_weights=need_weights,
            gauss_scale=self.gauss_scale,
            atten_scale=self.atten_scale,
            onehot=onehot,
            params=self.params,
            this_epoch=this_epoch)

        return attn_output, attn_output_weights

def scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        batch: Tensor,
        batch_kv: Tensor,
        edges: Tensor,
        attn_weights: Optional[Tensor] = None,
        values: Tensor = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        edges: index pairs (i,j) to define attentions between q and p,v.
        attn_weights: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(Nt, B, E)` where Nt is the target sequence length, B is batch size,
            and E is embedding dimension.
        - key: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - value: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - edges: :math:`(2, M)` where M is the edge num.
        - attn_weights: `(M, B)` where M in the edge num, B is batch size.
        - Output: attention values have shape :math:`(Nt, B, E)`; attention weights
            have shape :math:`(M, B)` where M in the edge num, B is batch size.
    """
    Nt, B, E = q.shape
    q = q / math.sqrt(E)
    # (M, B, E) x (M, B, E) -> (M, B)
    attn = (q[edges[0]] * k[edges[1]]).sum(dim=-1)

    # flag = torch.are_deterministic_algorithms_enabled()
    # torch.use_deterministic_algorithms(False)
    bsz = batch.max().item() + 1
    q_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
    q_sizes.scatter_add_(0, batch, torch.ones_like(batch))

    if batch_kv is batch:
        k_sizes = q_sizes
    else:
        k_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
        k_sizes.scatter_add_(0, batch_kv, torch.ones_like(batch_kv))
    # This is because self-attention has the same number of queries and keys (sys_size).
    edg_sizes = q_sizes * k_sizes

    q_sizes = q_sizes.tolist()
    k_sizes = k_sizes.tolist()
    edg_sizes = edg_sizes.tolist()
    # torch.use_deterministic_algorithms(flag)

    if True:
        # The scaled_dot operation involves the summations along the key axis
        # whose size varies among batch samples. So we split concatenated data
        # into a list of batch samples and apply the scaled_dot for each sample.
        # We could do the same without the splitting & looping by using scatter_add,
        # but we rather avoid scatter_add as it breaks reproducibility in backprop.
        if attn_weights is None:
            attn_weights = 0

        # standard normalization for all system points.
        attn += attn_weights
        attn = torch.split_with_sizes(attn, edg_sizes)
        attn = torch.cat(
            [F.softmax(a.view(qs, ks, -1), dim=1).view(qs * ks, -1) for a, qs, ks in zip(attn, q_sizes, k_sizes)])

        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)

        # (M, B, 1) x (M, B, E) -> (q_len, B, E)
        if values is None:
            output = attn[..., None] * v[edges[1]]
        else:
            output = attn[..., None] * (v[edges[1]] + values)
        output = torch.split_with_sizes(output, edg_sizes)
        output = torch.cat([o.view((qs, ks) + o.shape[1:]).sum(dim=1) for o, qs, ks in zip(output, q_sizes, k_sizes)])
    else:
        if attn_weights is not None:
            attn += attn_weights

        # This code was slower (3.65 it/sec vs 3.95 it/sec).
        attn = torch.split_with_sizes(attn, edg_sizes)
        v = torch.split_with_sizes(v, sys_sizes)
        output = []
        for a, v, s in zip(attn, v, sys_sizes):
            a = F.softmax(a.view(s, s, -1), dim=1)
            if dropout_p > 0.0:
                a = F.dropout(a, p=dropout_p)
            # (Nt,Nt,B)x(1,Nt,B,E).sum(dim=1) -> # (Nt,B,E)
            output.append((a[..., None] * v[None]).sum(dim=1))
        output = torch.cat(output)

    sizes = (q_sizes, k_sizes, edg_sizes)
    return output, attn, sizes


def _mha_shape_check(
        query: Tensor, key: Tensor, value: Tensor, edges: Tensor,
        dist2: Tensor, cos_kr: Tensor, k2: Tensor, vcell: Tensor, selfe: Tensor, num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `dist`, and `edges`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    assert query.dim() == 2, f"Expected `query` to be 2-D but got a {query.dim()}-D tensor."
    assert key.dim() == 2, f"Expected `key` to be 2-D but got a {key.dim()}-D tensor."
    assert value.dim() == 2, f"Expected `value` to be 2-D but got a {value.dim()}-D tensor."

    assert edges.dim() == 2, f"Expected `edges` to be 2-D but got a {edges.dim()}-D tensor."
    assert edges.shape[0] == 2
    assert edges.dtype == torch.long

    if dist2 is not None:
        assert dist2.dim() == 2, f"Expected `dist` to be 2-D but got a {dist2.dim()}-D tensor."
        assert edges.shape[1] == dist2.shape[0]

    if cos_kr is not None:
        assert cos_kr.dim() == 2, f"Expected `dist2` to be 2-D but got a {cos_kr.dim()}-D tensor."
        assert k2.dim() == 2, f"Expected `k2` to be 2-D but got a {k2.dim()}-D tensor."
        assert vcell.dim() == 1, f"Expected `vcell` to be 1-D but got a {vcell.dim()}-D tensor."
        assert cos_kr.shape[0] == edges.shape[1]
        assert k2.shape[0] == vcell.shape[0]
        assert k2.shape[1] == cos_kr.shape[1]
    if selfe is not None:
        assert selfe.dim() == 1, f"Expected `selfe` to be 1-D but got a {selfe.dim()}-D tensor."


class LogSumExpXYFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, dim, keepdim=False):
        # x: (edges,     1, mirrors)
        # y: (edges, heads,       1)
        # z = log(sum_i exp(xi*yi))

        # `x*y` invokes breadcasting that results in a huge tensor.
        # This custom function does not hold x*y for backward,
        # but just hold the inputs x and y as is, which does not increase memory usage.
        # TODO: write a custom cuda kernel to avoid temporary memory allocation for x*y.
        z = torch.logsumexp(x * y, dim=dim, keepdim=keepdim)

        ctx.save_for_backward(x, y, z)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return z

    @staticmethod
    def backward(ctx, gz):
        x, y, z = ctx.saved_tensors
        gx = gy = None

        # z = log(sum_i exp(xi*yi))
        # dz/dxj = yj*exp(xj*yj) / (sum_i exp(xi*yi))
        #        = yj*exp(xj*yj) / exp(z)
        #        = yj*exp(xj*yj - z)
        if not ctx.keepdim:
            z = z.unsqueeze(ctx.dim)
            gz = gz.unsqueeze(ctx.dim)
        g = gz * torch.exp(x * y - z)

        if ctx.needs_input_grad[0]:
            gx = y * g
            # if x was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i, (si, so) in enumerate(zip(gx.shape, x.shape)) if si != so]
            if len(dims) > 0:
                gx = gx.sum(dim=dims, keepdim=True)

        if ctx.needs_input_grad[1]:
            gy = x * g
            # if y was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i, (si, so) in enumerate(zip(gy.shape, y.shape)) if si != so]
            if len(dims) > 0:
                gy = gy.sum(dim=dims, keepdim=True)

        return gx, gy, None, None


class AverageBySoftmaxXYFunc_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, x, y, dim, W=None, keepdim=False):
        # a: (edges,     1, mirrors, K)
        # x: (edges,     1, mirrors, 1)
        # y: (edges, heads,       1, 1)
        # W: (heads, head_dim, K)
        # z = log(sum_i exp(xi*yi))

        # `x*y` invokes breadcasting that results in a huge tensor.
        # This custom function does not hold x*y for backward,
        # but just hold the inputs x and y as is, which does not increase memory usage.
        # TODO: write a custom cuda kernel to avoid temporary memory allocation for x*y.
        p = torch.softmax(x * y, dim=dim)
        # z = [(p[:, i:i+1]*a).sum(dim=dim, keepdim=keepdim) for i in range(p.shape[1])]
        # p: (edges, heads, mirrors, 1) - > (edges, heads, 1, mirrors)
        # a: (edges,     1, mirrors, K)
        E, H, R, _ = p.shape
        p = p.reshape(E, H, 1, R)
        # Below is equivalent to: z = p @ a
        # but unloop it along the heads axis to reduce the peak memory usage.
        # z: [edges, 1, 1, K] x heads
        z = [p[:, i:i + 1] @ a for i in range(H)]
        if W is not None:
            assert W.dim() in (2, 3, 4)
            W_ = W
            if W_.dim() == 2:
                W_ = W_.unsqueeze(0)
            if W_.dim() == 3:
                W_ = W_.unsqueeze(0)
            if W_.shape[1] == 1:
                W_ = W_.expand(W_.shape[0], H, W_.shape[-2], W_.shape[-1])
            W_ = W_.transpose(2, 3)

            z0 = torch.cat(z, dim=1)
            # [(edges, 1, 1, K) x (K, D)] = [(edges, 1, 1, D)]
            z = [z[i] @ W_[:, i:i + 1] for i in range(H)]
            del W_

        z = torch.cat(z, dim=1)
        if not keepdim:
            z.squeeze_(2)

        if W is None:
            ctx.save_for_backward(a, x, y)
        else:
            ctx.save_for_backward(a, x, y, W, z0)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return z

    @staticmethod
    def backward(ctx, gz):
        if len(ctx.saved_tensors) == 3:
            a, x, y = ctx.saved_tensors
            W = z0 = None
        else:
            a, x, y, W, z0 = ctx.saved_tensors
        ga = gx = gy = gW = None

        # zi = exp(xi*yi)*ai/(sum_k exp(xk*yk))
        #    = a*p where p = softmax(x*y).
        # dz/dxi = p*y*(a*g - (p*a*g).sum())
        if not ctx.keepdim:
            gz = gz.unsqueeze(ctx.dim)

        if W is not None:
            # gz: (E, H, 1, D) -> (E,H,K,D) -> (H,K,D)
            # W : (H, D, K)
            # z0: (E, H, 1, K)
            # gW = [(gz_*z0_).sum(0) for gz_, z0_ in zip(torch.split(gz.transpose(2,3), 1, 1), torch.split(z0, 1, 1))]
            # gW = torch.cat(gW, 0)
            # (H,1,D,E) x (H,1,E,K) -> (H,1,D,K)
            gW = gz.permute(1, 2, 3, 0) @ z0.permute(1, 2, 0, 3)
            gW.squeeze_(1)
            # gW = (gz.transpose(2,3)*z0).sum(0)

            # gz: (E,H,1,D)x(H,D,K) -> (E,H,1,K)
            gz0 = gz @ W  # Compute it later as this step is costly.
            # gz0 = None
        else:
            gz0 = gz

        # p: (E,H,R,1)
        p = torch.softmax(x * y, dim=ctx.dim)

        if ctx.needs_input_grad[0]:
            # ga: (E,1,R,K)
            ga = p * gz0 if gz0 is not None else p * (gz @ W)
            dims = [i for i, (si, so) in enumerate(zip(ga.shape, a.shape)) if si != so]
            if len(dims) > 0:
                ga = ga.sum(dim=dims, keepdim=True)

        if False:
            # This code requires huge temporary memory
            # for computing g*a of size (edges, heads, mirrors, K).
            # a: (edges,     1, mirrors, K)
            # x: (edges,     1, mirrors, 1)
            # y: (edges, heads,       1, 1)
            # z: (edges, heads,       1, K)
            g = a * gz0
            g -= (g * p).sum(dim=ctx.dim, keepdim=True)
            g *= p

        else:
            # a: (edges,     1, mirrors, K)
            # x: (edges,     1, mirrors, 1)
            # y: (edges, heads,       1, 1)
            # z: (edges, heads,       1, K)

            # Unloop the last axis (size K)
            K = a.shape[-1]
            s = torch.broadcast_shapes(a.shape[:3], gz.shape[:3])
            g = torch.zeros(s, device=gz.device, dtype=gz.dtype)
            if gz0 is not None:
                for i in range(K):
                    g += a[..., i] * gz0[..., i]
            else:
                # Because gz has not been updated as gz@W, compute it here in the loop.
                # gz@W : (E,H,1,D)x(H,D,K)
                for i in range(K):
                    g += a[..., i] * (gz @ W[..., i:i + 1]).squeeze(-1)
            g.unsqueeze_(-1)
            g -= (g * p).sum(dim=ctx.dim, keepdim=True)
            g *= p
            del p, gz, gz0

        # gx = y*p*(a*gz - (a*gz*p).sum())
        if ctx.needs_input_grad[1]:
            gx = y * g
            # if x was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i, (si, so) in enumerate(zip(gx.shape, x.shape)) if si != so]
            if len(dims) > 0:
                gx = gx.sum(dim=dims, keepdim=True)

        # gy = x*p*(a*gz - (a*gz*p).sum())
        # (E,H,1,1) = (E,1,R,1)*(E,H,R,1)*(E,H,R,1)
        if ctx.needs_input_grad[2]:
            gy = x * g
            # if y was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i, (si, so) in enumerate(zip(gy.shape, y.shape)) if si != so]
            if len(dims) > 0:
                gy = gy.sum(dim=dims, keepdim=True)

        return ga, gx, gy, None, gW, None

def _get_basis_func(m, K, x, method, width_scale=1.0):
    if method is None or method in ("", "cgcnn"):
        a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
        a = a - x**0.5
        a /= ((2**0.5)*(width_scale*m/K))
        a *= a
        torch.neg_(a)
        torch.exp_(a)
        return a
    if method == "physnet":
        # vij_k = exp (−βk (exp(−rij ) − µk)**2)
        # βk: a fixed βk = [2 (1 − exp(−rcut))/K ]**−2
        # µk: equally spaced between exp(−rcut) and 1 with rcut = 10
        a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
        w = (1.0-math.exp(-m))/K
        beta = (2.0*w)**-2


class AverageGaussiansBySoftmaxXYFunc_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, K, x, y, dim, W=None, width_scale=1.0, basis_func: str = "", keepdim=False):
        # a: (edges,     1, mirrors, K)
        # x: (edges,     1, mirrors, 1)
        # y: (edges, heads,       1, 1)
        # W: (edges, heads, head_dim, K)
        # z = log(sum_i exp(xi*yi))

        # `x*y` invokes breadcasting that results in a huge tensor.
        # This custom function does not hold x*y for backward,
        # but just hold the inputs x and y as is, which does not increase memory usage.
        # TODO: write a custom cuda kernel to avoid temporary memory allocation for x*y.
        p = torch.softmax(x * y, dim=dim)
        # z = [(p[:, i:i+1]*a).sum(dim=dim, keepdim=keepdim) for i in range(p.shape[1])]
        # p: (edges, heads, mirrors, 1) - > (edges, heads, 1, mirrors)
        # a: (edges,     1, mirrors, K)
        E, H, R, _ = p.shape
        p = p.reshape(E, H, 1, R)

        # a = exp((mu_k - dist)^2/(2*mu_0^2))
        a = _get_basis_func(m, K, x, basis_func, width_scale)
        # a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
        # a = a - x**0.5
        # a /= ((2**0.5)*(m/K))
        # a *= a
        # torch.neg_(a)
        # torch.exp_(a)

        # Below is equivalent to: z = p @ a
        # but unloop it along the heads axis to reduce the peak memory usage.
        # z: [edges, 1, 1, K] x heads
        z = [p[:, i:i + 1] @ a for i in range(H)]
        # try:
        #     z = p @ a
        # except:
        #     z = [p[:, i:i+1] @ a for i in range(H)]
        del a

        if W is not None:
            assert W.dim() in (2, 3, 4)
            W_ = W
            if W_.dim() == 2:
                W_ = W_.unsqueeze(0)
            if W_.dim() == 3:
                W_ = W_.unsqueeze(0)
            if W_.shape[0] == 1:
                W_ = W_.expand(W_.shape[0], H, W_.shape[-2], W_.shape[-1])
            W_ = W_.transpose(-2, -1)

            z0 = torch.cat(z, dim=1)
            z = [z[i] @ W_[:, i:i + 1] for i in range(H)]
            # try:
            #     z = z0 @ W_
            # except:
            #     # [(edges, 1, 1, K) x (K, D)] = [(edges, 1, 1, D)]
            #     z = [z0[:, i:i+1] @ W_[:, i:i+1] for i in range(H)]
            del W_

        z = torch.cat(z, dim=1)

        if not keepdim:
            z.squeeze_(2)

        if W is None:
            ctx.save_for_backward(x, y)
        else:
            ctx.save_for_backward(x, y, W, z0)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.K = K
        ctx.m = m
        ctx.basis_func = basis_func
        ctx.width_scale = width_scale
        return z

    @staticmethod
    def backward(ctx, gz):
        if len(ctx.saved_tensors) == 2:
            x, y = ctx.saved_tensors
            W = z0 = None
        else:
            x, y, W, z0 = ctx.saved_tensors
        gx = gy = gW = None
        m = ctx.m
        K = ctx.K
        basis_func = ctx.basis_func
        width_scale = ctx.width_scale

        # zi = exp(xi*yi)*ai/(sum_k exp(xk*yk))
        #    = a*p where p = softmax(x*y).
        # dz/dxi = p*y*(a*g - (p*a*g).sum())
        if not ctx.keepdim:
            gz = gz.unsqueeze(ctx.dim)

        if W is not None:
            # gz: (E, H, 1, D) -> (E,H,K,D) -> (H,K,D)
            # W : (E, H, D, K)
            # z0: (E, H, 1, K)
            # gW = [(gz_*z0_).sum(0) for gz_, z0_ in zip(torch.split(gz.transpose(2,3), 1, 1), torch.split(z0, 1, 1))]
            # gW = torch.cat(gW, 0)
            # (H,1,D,E) x (H,1,E,K) -> (H,1,D,K)
            if W.shape[0] == 1:
                gW = gz.permute(1, 2, 3, 0) @ z0.permute(1, 2, 0, 3)
                gW.transpose_(0, 1)
            else:
                gW = gz.transpose(-1, -2) * z0

            # gW = (gz.transpose(2,3)*z0).sum(0)

            # gz: (E,H,1,D)x(E,H,D,K) -> (E,H,1,K)
            try:
                gz0 = gz @ W
            except:
                # Compute later if fails due to memory shortage.
                gz0 = None
        else:
            gz0 = gz

        # p: (E,H,R,1)
        p = torch.softmax(x * y, dim=ctx.dim)

        if True:
            a = _get_basis_func(m, K, x, basis_func, width_scale)
            # a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
            # a = a - x**0.5
            # a /= ((2**0.5)*m/K)
            # a *= a
            # torch.neg_(a)
            # torch.exp_(a)

            # a: (edges,     1, mirrors, K)
            # x: (edges,     1, mirrors, 1)
            # y: (edges, heads,       1, 1)
            # z: (edges, heads,       1, K)

            # Unloop the last axis (size K) of a
            s = torch.broadcast_shapes(a.shape[:3], gz.shape[:3])
            g = torch.zeros(s, device=gz.device, dtype=gz.dtype)
            if gz0 is not None:
                for i in range(K):
                    g += a[..., i] * gz0[..., i]
            else:
                # Because gz has not been updated as gz@W, compute it here in the loop.
                # gz@W : (E,H,1,D)x(H,D,K)
                for i in range(K):
                    g += a[..., i] * (gz @ W[..., i:i + 1]).squeeze(-1)
            del a
            g.unsqueeze_(-1)
            g -= (g * p).sum(dim=ctx.dim, keepdim=True)
            g *= p
            del p, gz, gz0

        # gx = y*p*(a*gz - (a*gz*p).sum())
        if ctx.needs_input_grad[2]:
            gx = y * g
            # if x was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i, (si, so) in enumerate(zip(gx.shape, x.shape)) if si != so]
            if len(dims) > 0:
                gx = gx.sum(dim=dims, keepdim=True)

        # gy = x*p*(a*gz - (a*gz*p).sum())
        # (E,H,1,1) = (E,1,R,1)*(E,H,R,1)*(E,H,R,1)
        if ctx.needs_input_grad[3]:
            gy = x * g
            # if y was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i, (si, so) in enumerate(zip(gy.shape, y.shape)) if si != so]
            if len(dims) > 0:
                gy = gy.sum(dim=dims, keepdim=True)

        return None, None, gx, gy, None, gW, None, None, None


def compute_real_domain_attn(alpha, dist2, edges0):
    # alpha: (nodes, heads)
    # dist2: (edge_num, R)
    # pe_wave: (edge_num, R, K)
    # where R = (1+2r)^3 and K is the PE dim.
    attn_weights = LogSumExpXYFunc.apply(
        dist2.unsqueeze(1),
        alpha.unsqueeze(2)[edges0],
        -1
    )
    return attn_weights


def _compute_reci_domain_attn(alpha, cos_kr, k2, vcell, edges0, batch):
    # We compute: sum k'[exp(-k^2/4a^2) * cos(kr)] for the real-domain func exp(-a^2*r^2).
    # Here, 1/a^2 = alpha in our code. So,
    #   sum k'[exp(-alpha^2*k^2/4) * cos(kr)]
    # cos_kr: (E, R)
    # k2    : (N, R)
    # alpha : (L, H)
    # vcell : (N)

    # self_coef = vcell.unsqueeze(1)[batch] / (2.0*math.pi*alpha)**(3/2)   # (L, H)
    alp = alpha.unsqueeze(2)  # (L, H, 1)
    k2 = k2.unsqueeze(1)[batch]  # (L, 1, R)
    cos_kr = cos_kr.unsqueeze(1)  # (E, 1, R)

    # NOTE: max alpha_k2 = 0 so exp(alpha_k2) does not overflow.
    alpha_k2 = alp * k2
    attn_weights = (torch.exp(alpha_k2)[edges0] * cos_kr).sum(dim=-1)
    attn_weights = torch.log(attn_weights.clamp_(1e-6))

    # NOTE: the following correction is required when using non-softmax attention.
    # In the paper, the correction term is a factor: (1/V)(2*pi/gamma)^(3/2).
    # Here, alpha = -1/(2*gamma). So, (2*pi/gamma)^3/2 = (-4*pi*alpha)^3/2.
    log_ci = (3 / 2) * (torch.log((-4 * math.pi) * alpha)) - torch.log(vcell)[batch, None]
    attn_weights += log_ci[edges0]

    return attn_weights

def indexed_lattice_multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        batch_q: Tensor,
        batch_kv: Tensor,
        edges: Tensor,
        dist2: Tensor,
        cos_kr: Tensor,
        k2: Tensor,
        vcell: Tensor,
        selfe: Tensor,
        pe_wave: Tensor,
        lattice_pos_weights: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        q_proj_weight: Tensor,
        k_proj_weight: Tensor,
        v_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        pe_dist_proj: Optional[Tensor],
        pe_wave_proj: Optional[Tensor],
        training: bool = True,
        need_weights: bool = True,
        gauss_scale: Optional[Tensor] = None,
        atten_scale: Optional[Tensor] = None,
        onehot: Optional[Tensor] = None,
        lattice_params: LatticeformerParams = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        dist: distance matrices of points of lattices.
        lattice_pos_weights: weights for lattice position embeddings.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - dist: :math:`(N, L, S, R)` or `(L, S, R)`, where N is the batch size, S is the source sequence length,
           N is the batch size, R is the number of neighbors of the lattice.
        - lattice_pos_weights: :math:`(E)`, where E is the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    _mha_shape_check(query, key, value, edges, dist2, cos_kr, k2, vcell, selfe, num_heads)

    # set up shape vars
    tgt_len, embed_dim = query.shape
    src_len, _ = key.shape
    esz = edges.shape[1]
    # assert embed_dim == embed_dim_to_check, \
    #     f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    # if isinstance(embed_dim, torch.Tensor):
    #     # embed_dim can be a tensor when JIT tracing
    #     head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    # else:
    #     head_dim = embed_dim // num_heads
    # assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    head_dim_q = q_proj_weight.shape[0] // num_heads
    head_dim_k = k_proj_weight.shape[0] // num_heads
    head_dim_v = v_proj_weight.shape[0] // num_heads
    assert head_dim_q * num_heads == q_proj_weight.shape[0]
    assert head_dim_k * num_heads == k_proj_weight.shape[0]
    assert head_dim_v * num_heads == v_proj_weight.shape[0]

    # allow MHA to have different embedding dimensions when separate projection weights are used
    assert key.shape[0] == value.shape[0], \
        f"key's sequence and batch dims {key.shape[0]} do not match value's {value.shape[0]}"

    #
    # compute in-projection
    #
    assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
    assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
    assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
    if in_proj_bias is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = in_proj_bias.split([q_proj_weight.size(0), k_proj_weight.size(0), v_proj_weight.size(0)])
    q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    #
    # reshape q, k, v for multihead attention
    #
    q = q.contiguous().view(tgt_len, num_heads, head_dim_q)
    k = k.contiguous().view(k.shape[0], num_heads, head_dim_k)
    v = v.contiguous().view(v.shape[0], num_heads, head_dim_v)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # compute attn_weights according to dist
    # TODO: this stes is the largest bottleneck of latticeformer.
    # dist: (esz, neighbors)
    # q: (tgt_len, num_heads, head_dim)
    # lattice_pos_weights: (num_heads*head_dim)
    if lattice_params is None or lattice_params.domain == "no":
        attn_weights = None
    else:
        domain = lattice_params.domain
        gauss_lb_real = lattice_params.gauss_lb_real
        gauss_lb_reci = lattice_params.gauss_lb_reci
        positive_func = lattice_params.positive_func

        # exp( -pos_func(alpha)*distance^2 )
        if lattice_params.gauss_state.startswith("q"):
            alpha = q.reshape(tgt_len, num_heads, 1, head_dim_q) @ lattice_pos_weights.reshape(num_heads, head_dim_q, 1)
        elif lattice_params.gauss_state == "1":
            ones = torch.ones_like(q)
            alpha = ones.reshape(tgt_len, num_heads, 1, head_dim_q) @ lattice_pos_weights.reshape(num_heads, head_dim_q,
                                                                                                  1)
        else:
            # query is original state 'x' before q = Wx
            # lattice_pos_weights's shape is (num_heads, embed_dim)
            alpha = F.linear(query, lattice_pos_weights)

        alpha = alpha.view(tgt_len, num_heads)
        if lattice_params.normalize_gauss:
            scale_available = gauss_scale[0] > 0
            if scale_available:
                scale = gauss_scale[0]
                shift = gauss_scale[1]
            else:
                scale = torch.reciprocal(alpha.detach().std())
                shift = -alpha.detach().mean()
                # print("Computed gauss scale and shift:", scale, shift)
            alpha += shift
            alpha *= scale

            # save the scale only once in the first training step.
            if training and not scale_available:
                torch.nn.init.constant_(gauss_scale[0], scale)
                torch.nn.init.constant_(gauss_scale[1], shift)
                print("Saved gauss scale and shift:", gauss_scale.data)

        if positive_func == 'abs':
            func = lambda x, lb: (1 - lb) * x.abs() + lb
        elif positive_func.startswith('softplus'):
            beta = float(positive_func.split('=')[1]) if '=' in positive_func else 1.0
            func = lambda x, lb: F.softplus(x + 1 / beta * math.log(math.exp(beta * (1 - lb)) - 1), beta=beta) + lb
        elif positive_func.startswith('exp'):
            beta = float(positive_func.split('=')[1]) if '=' in positive_func else 1.0
            func = lambda x, lb: (1.0 - lb) * torch.exp(x * (beta / (1.0 - lb))) + lb
        elif positive_func.startswith('elu'):
            beta = float(positive_func.split('=')[1]) if '=' in positive_func else 1.0
            func = lambda x, lb: (1.0 - lb) * F.elu(x * (beta / (1.0 - lb))) + 1.0
        elif positive_func.startswith('sigmoid'):
            beta = float(positive_func.split('=')[1]) if '=' in positive_func else 1.0
            func = lambda x, lb: (1 / lb - lb) * F.sigmoid(x * (beta * (1 + lb) / (1.0 - lb)) + math.log(lb)) + lb
        else:
            raise NotImplementedError(f'Unkown positive_func: {positive_func}')

        # a = alpha
        # c = func(alpha, gauss_lb_real)
        # print("layer\t{:.4f}\t{:.4f}\t{:.4f} | {:.4f}\t{:.4f}\t{:.4f}" .format(a.mean(), a.median(), a.std(), c.mean(), c.median(), c.std()))

        '''
        Saved gauss scale and shift: tensor([ 2.5387e+01, -6.5168e-03], device='cuda:0')
        Saved gauss scale and shift: tensor([3.1654e+01, 5.2495e-04], device='cuda:0')
        Saved gauss scale and shift: tensor([3.0025e+01, 7.2772e-03], device='cuda:0')
        Saved gauss scale and shift: tensor([2.2007e+01, 3.1712e-03], device='cuda:0')
        '''
        if domain == "real":
            alpha = func(alpha, gauss_lb_real) * (-0.5 * lattice_params.scale_real ** -2)
            attn_weights = compute_real_domain_attn(alpha, dist2, edges[0])
        elif domain == "reci":
            alpha = func(alpha, gauss_lb_reci) * (-0.5 * lattice_params.scale_reci ** 2)
            attn_weights =  _compute_reci_domain_attn(alpha, cos_kr, k2, vcell, edges[0], batch_q)
        elif domain == "RECI":
            # NOTE: for func_test, use this version.
            alpha = 1 / func(alpha, gauss_lb_reci) * (-0.5 * lattice_params.scale_reci ** 2)
            attn_weights = _compute_reci_domain_attn(alpha, cos_kr, k2, vcell, edges[0], batch_q)
        elif domain == "multihead":
            a1, a2 = alpha.chunk(2, 1)
            a1 = func(a1, gauss_lb_real) * (-0.5 * lattice_params.scale_real ** -2)
            a2 = func(a2, gauss_lb_reci) * (-0.5 * lattice_params.scale_reci ** 2)
            attn_weights = torch.cat([
                compute_real_domain_attn(a1, dist2, edges[0]),
                _compute_reci_domain_attn(a2, cos_kr, k2, vcell, edges[0], batch_q)
            ], dim=-1)
            alpha = torch.cat([a1, a2], dim=1)
        else:
            raise NotImplementedError(f"Not implmeneted for domain = {domain}.")

    # softmax(dist2*alpha)*pe_wave
    # (edges, 1, R, 1)*(edges, heads, 1, 1)*(edges, 1, R, K)
    values = None
    if pe_wave_proj is not None and domain == "real":
        values = AverageBySoftmaxXYFunc_.apply(
            pe_wave[:, None, :, :],
            dist2[:, None, :, None],
            alpha[:, :, None, None][edges[0]],
            -2,
            pe_wave_proj
        )
    elif pe_wave_proj is not None and domain in ["reci", "RECI"]:
        values = AverageBySoftmaxXYFunc_.apply(
            pe_wave[:, None, :, :],
            dist2[:, None, :, None],
            alpha[:, :, None, None][edges[0]],
            -2,
            pe_wave_proj
        )
        # (edges, heads, 1, D) =
        # (edges, heads, 1, K) x (heads, K, D)
        # V = values.squeeze(2)
        # V = torch.stack([V[:,i] @ pe.t() for i,pe in enumerate(pe_wave_proj.chunk(num_heads, 0))], 1)
        # values = V

    if pe_dist_proj is not None and domain in "real":
        dist_max = lattice_params.value_pe_dist_max
        if dist_max < 0:
            dist_max = (-dist_max) * lattice_params.scale_real
        values = AverageGaussiansBySoftmaxXYFunc_.apply(
            dist_max,
            lattice_params.value_pe_dist_real,
            dist2[:, None, :, None],
            alpha[:, :, None, None][edges[0]],
            -2,
            pe_dist_proj,
            lattice_params.value_pe_width_scale
        )
    elif pe_dist_proj is not None and domain in ["reci", "RECI"]:
        dist_max = lattice_params.value_pe_dist_max
        if dist_max < 0:
            dist_max = (-dist_max) * lattice_params.scale_real
        values = AverageGaussiansBySoftmaxXYFunc_.apply(
            dist_max,
            lattice_params.value_pe_dist_reci,
            dist2[:, None, :, None],
            alpha[:, :, None, None][edges[0]],
            -2,
            pe_dist_proj,
            lattice_params.value_pe_dist_wscale
        )
        # values = values @ pe_wave_proj.view(num_heads, head_dim, -1).transpose(1,2)
        # values = values.view(-1, num_heads, head_dim)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights, sizes = scaled_dot_product_attention(q, k, v, batch_q, batch_kv, edges,
                                                                            attn_weights, values, dropout_p)

    attn_output = attn_output.contiguous().view(tgt_len, head_dim_v * num_heads)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(esz, num_heads)
        attn_output_weights = attn_output_weights.mean(dim=1)
        return attn_output, attn_output_weights
    else:
        return attn_output, values

class IndexedLatticeMultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 kdim=None, vdim=None,
                 params=LatticeformerParams(),
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(IndexedLatticeMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim * num_heads if kdim is not None and kdim > 0 else embed_dim
        self.vdim = vdim * num_heads if vdim is not None and vdim > 0 else embed_dim
        assert params.domain in ("real", "reci", "RECI", "multihead", "no")
        if params.domain == "multihead":
            assert num_heads % 2 == 0, "In multihead mode, num_head must be even as MHA of each domain uses num_head/2 of heads."

        # lattice parameters that are referenced in indexed_lattice_multi_head_attention_forward.
        self.params = params
        self.gauss_scale = Parameter(torch.zeros(2, **factory_kwargs), requires_grad=False)
        self.atten_scale = Parameter(torch.zeros(1, **factory_kwargs), requires_grad=False)

        self.num_heads = num_heads
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((self.vdim, embed_dim), **factory_kwargs))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(self.kdim * 2 + self.vdim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(self.vdim, embed_dim, bias=bias, **factory_kwargs)

        if params.domain == "no":
            self.lattice_pos_weights = None
        elif params.gauss_state.startswith("q"):
            self.lattice_pos_weights = Parameter(torch.empty((self.kdim), **factory_kwargs))
        elif params.gauss_state == "1":
            self.lattice_pos_weights = Parameter(torch.empty((self.kdim), **factory_kwargs))
        elif params.gauss_state == "x-xn":
            self.lattice_pos_weights = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
            self.lattice_pos_weights2 = Parameter(torch.empty((self.kdim), **factory_kwargs))
        elif params.gauss_state.startswith("x"):
            self.lattice_pos_weights = Parameter(torch.empty((self.num_heads, embed_dim), **factory_kwargs))

        self.ATOM_NUM = 98
        head = self.num_heads
        cond = 1

        if params.domain in ["real", "multihead"]:
            self.pe_dist_proj = Parameter(
                torch.empty(cond, head, self.head_dim, params.value_pe_dist_real, **factory_kwargs)) \
                if params.value_pe_dist_real > 0 else None
            self.pe_wave_proj = Parameter(
                torch.empty(cond, head, self.head_dim, params.value_pe_wave_real ** 3, **factory_kwargs)) \
                if params.value_pe_wave_real > 0 else None
        elif params.domain in ["reci", "RECI"]:
            self.pe_dist_proj = Parameter(
                torch.empty(cond, head, self.head_dim, params.value_pe_dist_reci, **factory_kwargs)) \
                if params.value_pe_dist_reci > 0 else None
            self.pe_wave_proj = Parameter(
                torch.empty(cond, head, self.head_dim, params.value_pe_wave_reci ** 3, **factory_kwargs)) \
                if params.value_pe_wave_reci > 0 else None

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.out_proj.weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

        # TODO: consider the normalization method later
        if self.lattice_pos_weights is not None:
            if self.params.gauss_state in ("q", "q-headdim", "1"):
                normal_(self.lattice_pos_weights, 0., (self.kdim // self.num_heads) ** -0.5)
            elif self.params.gauss_state == "q-kdim":
                normal_(self.lattice_pos_weights, 0., (self.kdim) ** -0.5)
            elif self.params.gauss_state in ("x", "x-norm"):
                normal_(self.lattice_pos_weights, 0., (self.embed_dim) ** -0.5)
            elif self.params.gauss_state == "x-xavier":
                xavier_uniform_(self.lattice_pos_weights)
            elif self.params.gauss_state == "x-xn":
                xavier_uniform_(self.lattice_pos_weights)
                normal_(self.lattice_pos_weights2, 0., (self.kdim // self.num_heads) ** -0.5)
            elif self.params.gauss_state == "x-xn2":
                H, K, D = self.num_heads, self.kdim, self.embed_dim
                W = self.lattice_pos_weights
                W1 = torch.empty((K, D), device=W.device, dtype=W.dtype)
                W2 = torch.empty((H, 1, K // H), device=W.device, dtype=W.dtype)
                xavier_uniform_(W1)
                normal_(W2, 0., (K // H) ** -0.5)
                W0 = W2 @ W1.reshape((H, K // H, D))
                with torch.no_grad():
                    self.lattice_pos_weights.set_(W0.reshape_as(W))
            else:
                raise NotImplementedError()

        # D0 = 1/sqrt(I + O)
        # D1 = 1/sqrt(I + O*H)
        # I = 64, O = 16, H = 8
        # D0 = 1/sqrt(64 + 16) = 1/( 4*sqrt(5) )
        # D1 = 1/sqrt(64 + 16*8) = 1/( 8*sqrt(3) )
        # D1/D0 = 0.645497224
        # good case = 0.0099
        # current (no t-fixup)  = 0.0063
        if self.pe_dist_proj is not None:
            with torch.no_grad():
                for i, W in enumerate(self.pe_dist_proj):
                    W = W.view(-1, W.shape[-1])
                    co, ci = W.shape
                    a = (self.embed_dim + ci) / (co + ci)
                    # 'a' is to keep the scale regardless of pe_headed True or False.
                    xavier_uniform_(W, (ci) ** -0.5)

        if self.pe_wave_proj is not None:
            with torch.no_grad():
                for i, W in enumerate(self.pe_wave_proj):
                    W = W.view(-1, W.shape[-1])
                    co, ci = W.shape
                    a = (self.embed_dim + ci) / (co + ci)
                    # 'a' is to keep the scale regardless of pe_headed True or False.
                    xavier_uniform_(W, (ci) ** -0.5)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, batch_q: Tensor, batch_kv: Tensor, edges: Tensor,
                dist2: Tensor = None,
                cos_kr: Tensor = None, k2: Tensor = None, vcell: Tensor = None, selfe: Tensor = None,
                pe_wave: Tensor = None,
                onehot: Tensor = None,
                need_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """

        W = self.lattice_pos_weights
        if self.params.gauss_state == "x-xn":
            H, K, D = self.num_heads, self.kdim, self.embed_dim
            W2 = self.lattice_pos_weights2.view(H, 1, K // H)
            W = W2 @ W.reshape((H, K // H, D))
            W = W.reshape(H, D)

        attn_output, attn_output_weights = indexed_lattice_multi_head_attention_forward(
            query, key, value, batch_q, batch_kv, edges,
            dist2,
            cos_kr, k2, vcell, selfe, pe_wave,
            W,
            self.embed_dim, self.num_heads,
            self.q_proj_weight, self.k_proj_weight, self.v_proj_weight,
            self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            self.pe_dist_proj,
            self.pe_wave_proj,
            training=self.training,
            need_weights=need_weights,
            gauss_scale=self.gauss_scale,
            atten_scale=self.atten_scale,
            onehot=onehot,
            lattice_params=self.params)
        return attn_output, attn_output_weights

def get_edge_index(sizes, device):
    edges_i = []
    edges_j = []

    factory_kywd_long = {'dtype': torch.long, 'device': device}
    cur_index = 0
    sizes = sizes.tolist() if torch.is_tensor(sizes) else sizes
    for i, num in enumerate(sizes):
        inds = torch.arange(cur_index, cur_index + num, **factory_kywd_long)
        inds_i = inds.reshape(num, 1).repeat(1, num).flatten()
        inds_j = inds.reshape(1, num).repeat(num, 1).flatten()
        edges_i.append(inds_i)
        edges_j.append(inds_j)
        cur_index += num
    edges_i = torch.cat(edges_i)
    edges_j = torch.cat(edges_j)
    edges = torch.stack((edges_i, edges_j))
    return edges.contiguous()


def det_3x3(m):
    #   0 1 2
    # 0 * * *
    # 1 * * *
    # 2 * * *
    return m[:, 0, 0] * m[:, 1, 1] * m[:, 2, 2] \
        + m[:, 0, 1] * m[:, 1, 2] * m[:, 2, 0] \
        + m[:, 0, 2] * m[:, 1, 0] * m[:, 2, 1] \
        - m[:, 0, 2] * m[:, 1, 1] * m[:, 2, 0] \
        - m[:, 0, 0] * m[:, 1, 2] * m[:, 2, 1] \
        - m[:, 0, 1] * m[:, 1, 0] * m[:, 2, 2]


def compute_lattice_distances(
        pos, batch, trans_vec, sizes, lattice_range,
        output_dis=True,
        output_rec=False,
        dim_pe_wave=0):
    # pos: (L,D)
    # batch: (L)
    # trans_vec: (N,D,D)
    # range: int

    D = pos.shape[-1]
    factory_kywd = {'dtype': pos.dtype, 'device': pos.device}
    factory_kywd_long = {'dtype': torch.long, 'device': pos.device}

    # split flat-batched data
    if sizes is None:
        sizes = torch.zeros(trans_vec.shape[0], **factory_kywd_long)
        sizes.scatter_add_(0, batch, torch.ones_like(batch))
    if torch.is_tensor(sizes):
        sizes = sizes.tolist()
    edges = get_edge_index(sizes, pos.device)

    # define the range of lattice to be considered
    grids = torch.arange(-lattice_range, lattice_range + 1, **factory_kywd)
    grids = torch.stack(torch.meshgrid([grids] * D, indexing='ij'), dim=-1)
    # grids: (2*r+1,2*r+1,2*r+1,D)
    grids = grids.reshape(-1, D)
    # grids: ((2*R+1)^3, D) = (R, D)

    dis_pql = cos_kr = k2 = vcell = self_edges = None
    wk_pql = None

    # TODO: validate pe_wave: cos -> sin
    # vcell seems to be always positive for MP data.

    # Note torch.det sometimes yields nan when MAGMA solver is used.
    # To avoid it, call torch.backends.cuda.preferred_linalg_library("cusolver")
    # See also https://github.com/pytorch/pytorch/issues/73622
    # vcell = torch.det(trans_vec)

    # Further note: there was still an error saying
    # torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED, when calling `cusolverDnSgetrf`. This error may appear if the input matrix contains NaN.
    # So, the det is now implemented manually.
    vcell = det_3x3(trans_vec)
    recip_vec = torch.cat([
        torch.cross(trans_vec[:, 1:2], trans_vec[:, 2:3], dim=2),
        torch.cross(trans_vec[:, 2:3], trans_vec[:, 0:1], dim=2),
        torch.cross(trans_vec[:, 0:1], trans_vec[:, 1:2], dim=2),
    ], dim=1) * (2.0 * math.pi / vcell[:, None, None])

    b2e = batch[edges[0]]  # (N)[b2e] -> (E)
    pos_p_q = pos[edges[1]] - pos[edges[0]]  # (E,D)
    pos_lat = grids @ trans_vec  # (N,R,D)   = (  R,D)x(N, D,D)
    pos_pql = pos_p_q[:, None] + pos_lat[b2e]  # (E,R,D)   = (E, 1,D)+(E, R,D)
    del pos_lat

    if dim_pe_wave > 0:
        u = torch.arange(1, dim_pe_wave + 1, **factory_kywd)
        u = (2.0 * u - (dim_pe_wave + 1)) / (2.0 * dim_pe_wave)
        u = torch.stack(torch.broadcast_tensors(
            u[:, None, None],
            u[None, :, None],
            u[None, None, :]
        ), dim=-1).view(-1, 3)

        wk = u @ recip_vec  # (N,K,D)   = (  K,D)x(N, D,D)

        K = dim_pe_wave ** 3
        if False:
            wk_pql = torch.zeros(pos_pql.shape[:2] + (2 * K,), **factory_kywd)

            # because wk_pql is large, compute it with minimum temporary memory usage
            wk = wk[b2e]

            wk_pql[..., :K] += pos_pql[:, :, None, 0] * wk[:, None, :, 0]
            wk_pql[..., :K] += pos_pql[:, :, None, 1] * wk[:, None, :, 1]
            wk_pql[..., :K] += pos_pql[:, :, None, 2] * wk[:, None, :, 2]
            wk_pql[..., K:] = wk_pql[..., :K]
            # wk_pql[..., :K] = pos_pql @ wk.transpose(1,2)[b2e]            # (E,R,K) = (E,R,D) x (E,D,K)
            # wk_pql[..., K:] = wk_pql[..., :K]

            torch.cos_(wk_pql[:, :, :K])
            torch.sin_(wk_pql[:, :, K:])
        else:
            wk_pql = pos_pql @ wk.transpose(1, 2)[b2e]
            torch.sin_(wk_pql)
        del wk, u

    if output_dis:
        dis_pql = torch.norm(pos_pql, dim=-1)  # (E,R)

    if output_rec:
        half_grids = grids[grids.shape[0] // 2:]
        assert abs(half_grids + grids[0:grids.shape[0] // 2 + 1].flip(0)).sum() == 0

        # [<r,b1>, <r,b2>, <r,b3>]
        cos_kr = pos_p_q[:, None] @ recip_vec.transpose(1, 2)[b2e]  # (E,1,D)   = (E, 1,D)x(E, D,D)
        # <k,r> = n1*<r,b1> + n2*<r,b2> + n3*<r,b3>]
        cos_kr.squeeze_(1)  # (E,D)
        cos_kr = half_grids @ cos_kr[..., None]  # (E,R,1)   = (  R,D)x(E, D,1)
        cos_kr.squeeze_(2)  # (E,R)
        cos_kr.cos_()
        cos_kr[:, 1:] += cos_kr[:, 1:]

        k2 = recip_vec @ recip_vec.transpose(1, 2)  # (N,D,D)
        h = half_grids[:, :, None] @ half_grids[:, None, :]  # (R,D,D) = (R,D,1)x(R,1,D)
        k2 = (k2[:, None] * h[None]).sum(dim=(2, 3))  # (N,R)   = (N,R,D,D).sum(DD)
        del h

        # When torch.backends.cuda.matmul.allow_tf32 = True,
        # the computations of cos_kr and k2 become very sensitive to numerical errors.
        # The above code produced more accurate values than the code below.
        # when compared to double-based computations as reference.
        if False:
            # k = n1*b1 + n2*b2 + n3*b3
            k2 = half_grids @ recip_vec  # (N,R,D)   = (   R,D)x(N, D,D)

            cos_kr = k2[b2e] @ pos_p_q[..., None]  # (E,R,1)   = (E, R,D)x(E, D,1)
            cos_kr.squeeze_(2)  # (E,R)
            cos_kr.cos_()
            cos_kr[:, 1:] += cos_kr[:, 1:]

            # k2 = <k,k>
            k2 = k2[..., None, :] @ k2[..., :, None]  # (N,R,1,1) = (N,R, 1,D)x(N,R, D,1)
            k2 = k2.reshape(k2.shape[0], k2.shape[1])  # (N,R)

    return dis_pql, cos_kr, k2, vcell, self_edges, edges, wk_pql


def compute_lattice_distances_for_cuda(
        pos, batch, trans_vec, sizes, lattice_range, cutoff_radius=0,
        output_real=False,
        output_reci=False):
    # pos: (L,D)
    # batch: (L)
    # trans_vec: (N,D,D)
    # range: int

    D = pos.shape[-1]
    factory_kywd = {'dtype': pos.dtype, 'device': pos.device}
    factory_kywd_long = {'dtype': torch.long, 'device': pos.device}

    # split flat-batched data
    if sizes is None:
        sizes = torch.zeros(trans_vec.shape[0], **factory_kywd_long)
        sizes.scatter_add_(0, batch, torch.ones_like(batch))
    if torch.is_tensor(sizes):
        sizes = sizes.tolist()
    edges = get_edge_index(sizes, pos.device)

    pos_p_q = dist2_min = kr_base = recip_vec = vcell = None

    vcell = det_3x3(trans_vec)
    vcell = vcell.contiguous()
    recip_vec = torch.stack([
        torch.cross(trans_vec[:, 1], trans_vec[:, 2], dim=1),
        torch.cross(trans_vec[:, 2], trans_vec[:, 0], dim=1),
        torch.cross(trans_vec[:, 0], trans_vec[:, 1], dim=1),
    ], dim=1) * (2.0 * math.pi / vcell[:, None, None])
    recip_vec = recip_vec.contiguous()

    pos_p_q = pos[edges[1]] - pos[edges[0]]  # (E,D)
    pos_p_q = pos_p_q.contiguous()

    if output_real and cutoff_radius >= 0.0:
        # Compute the nearest-neighbor distance for each atom, given lattice range.
        def shortest_distance(r):
            grids = torch.arange(-r, r + 1, **factory_kywd)
            grids = torch.stack(torch.meshgrid([grids] * D, indexing='ij'), dim=-1)
            # grids: (2r+1,2r+1,2r+1,D)
            grids = grids.reshape(-1, D)
            # grids: ((2r+1)^3, D) = (R, D)

            b2e = batch[edges[0]]  # (B)[b2e] -> (E)
            lattice = grids @ trans_vec  # (B,R,D)   = (  R,D)x(B, D,D)
            pos_pql = pos_p_q[:, None] + lattice[b2e]  # (E,R,D)   = (E, 1,D)+(E, R,D)
            del lattice, b2e
            d2min = (pos_pql * pos_pql).sum(axis=2).min(axis=1)[0]  # (E)
            d2min = d2min.contiguous()
            return d2min

        dist2_min = None
        # try:
        #     try:
        #         from .cuda_funcs.minimum_distance import compute_minimum_distance
        #         dist2_min = compute_minimum_distance(pos_p_q, trans_vec, batch, edges, torch.norm(recip_vec, 2, -1), cutoff_radius)
        #     except:
        #         if cutoff_radius == 0.0:
        #             dist2_min = shortest_distance(lattice_range)
        # except:
        #     dist2_min = None

    if output_reci:
        b2e = batch[edges[0]]  # (N)[b2e] -> (E)
        # [<r,b1>, <r,b2>, <r,b3>]
        kr_base = recip_vec[b2e] @ pos_p_q[..., None]  # (E,D,1)   = (E, D,D)x(E, D,1)
        kr_base.squeeze_(2)  # (E,D)
        kr_base = kr_base.contiguous()

    return pos_p_q, dist2_min, kr_base, recip_vec, vcell, edges

class IndexedLatticeformerEncoder(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        t_fixup_init: if ``True``, use the initialization scheme proposed by Huang et al. 2020 in
            "Improving Transformer Optimization Through Better Initialization". Default: ``False``.
        no_layer_norm: if ``True`` all the layer norm layers in the module are removed.
            Should be used with t_fixup_init=True. Default: ``False``.
    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 t_fixup_init=False, no_layer_norm=False,
                 lattice_params: LatticeformerParams = LatticeformerParams(),
                 k_dim=0,
                 v_dim=0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        factory_kwargs = {}
        super(IndexedLatticeformerEncoder, self).__init__()
        assert lattice_params.domain in (
        "real", "reci", "RECI", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real")
        self.lattice_params = lattice_params

        self.layers = ModuleList([
            IndexedLatticeformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                             activation, layer_norm_eps, norm_first,
                                             no_layer_norm,
                                             params=lattice_params.getLayerParameters(i),
                                             k_dim=k_dim,
                                             v_dim=v_dim,
                                             **factory_kwargs)
            for i in range(num_encoder_layers)
        ])
        if t_fixup_init:
            for layer in self.layers:
                layer.fixup_initialization(num_encoder_layers)

        self.num_layers = num_encoder_layers
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else None

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, pos: Tensor, batch: Tensor, trans: Tensor, sizes: Tensor, onehot: Tensor = None,
                this_epoch: int = None) -> Tensor:
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            batch: batch indices of the sequence (required).
            dist: the distances of lattice points to the encoder (required).
            edges: attention edges (required).
        Shape:
            - src: :math:`(S, E)`.
            - batch: `(S)`.
            - dist: :math:`(M, R)`, where M is the number of edges.
            - edges: :math:`(2, M)`.
            - output: :math:`(T, E)`.
            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        Examples:
            >>> output = transformer_model(src, src_mask=src_mask)
        """

        #
        batch = batch - batch[0]
        dist2, cos_kr, k2, vcell, selfe, edges, pe_wave = \
            compute_lattice_distances(
                pos, batch, trans, sizes,
                self.lattice_params.lattice_range,
                output_dis=self.lattice_params.domain in (
                "real", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real"),
                output_rec=self.lattice_params.domain in (
                "reci", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real", "RECI"),
                dim_pe_wave=max(self.lattice_params.value_pe_wave_real, self.lattice_params.value_pe_wave_reci),
            )
        if k2 is not None and torch.isnan(k2).any():
            print("nan in k2")

        # exp(-a^2 r^2)
        if dist2 is not None:
            # dist2 /= self.scale_real
            dist2 *= dist2
        if k2 is not None:
            # k2 *= (self.scale_reci**2)
            # vcell /= (self.scale_reci**3)
            pass

        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        output = src

        for mod in self.layers:
            output = mod(output, batch, edges, dist2, cos_kr, k2, vcell, selfe, pe_wave, onehot)

        if self.norm is not None:
            output = self.norm(output)

        return output

class IndexedLatticeformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 no_layer_norm=False,
                 params=LatticeformerParams(),
                 k_dim=0,
                 v_dim=0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        factory_kwargs = {}

        assert params.domain in ("real", "reci", "RECI", "multihead")
        self.domain = params.domain

        super(IndexedLatticeformerEncoderLayer, self).__init__()
        self.self_attn = IndexedLatticeMultiheadAttention(
            d_model, nhead, dropout=dropout,
            kdim=k_dim,
            vdim=v_dim,
            params=params,
            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout) if dropout > 0 else lambda x: x
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else (lambda x: x)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else (lambda x: x)
        self.dropout1 = Dropout(dropout) if dropout > 0 else lambda x: x
        self.dropout2 = Dropout(dropout) if dropout > 0 else lambda x: x

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.add_zero_attn = False  # add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            constant_(self.linear1.bias, 0)
        if self.linear2.bias is not None:
            constant_(self.linear2.bias, 0)

    def fixup_initialization(self, num_layers):
        temp_state_dic = {}
        en_layers = num_layers

        for name, param in self.named_parameters():
            if name in ["linear1.weight",
                        "linear2.weight",
                        "self_attn.out_proj.weight",
                        "self_attn.v_proj_weight",
                        # "self_attn.pe_dist_proj",
                        # "self_attn.pe_wave_proj",
                        ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(IndexedLatticeformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, batch: Tensor, edges: Tensor,
                dist2: Tensor = None,
                cos_kr: Tensor = None, k2: Tensor = None, vcell: Tensor = None, selfe: Tensor = None,
                pe_wave: Tensor = None,
                onehot: Tensor = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            batch: batch indices of the sequence (required).
            dist: the distances of lattice points to modulate attentions (required).
            edges: the attention edges (required).
        Shape:
            see the docs in Transformer class.
        """
        if self.domain in ("real", "multihead"):
            assert dist2 is not None

        if self.domain in ("reci", "multihead", "RECI"):
            assert cos_kr is not None and k2 is not None and vcell is not None

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), batch, edges, dist2, cos_kr, k2, vcell, selfe, pe_wave, onehot)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, batch, edges, dist2, cos_kr, k2, vcell, selfe, pe_wave, onehot))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, batch: Tensor, edges: Tensor,
                  dist2: Tensor = None,
                  cos_kr: Tensor = None, k2: Tensor = None, vcell: Tensor = None, selfe: Tensor = None,
                  pe_wave: Tensor = None, onehot: Tensor = None) -> Tensor:
        x = self.self_attn(x, x, x, batch, batch, edges, dist2, cos_kr, k2, vcell, selfe, pe_wave, onehot,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class CrystalformerEncoderCUDA(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        t_fixup_init: if ``True``, use the initialization scheme proposed by Huang et al. 2020 in
            "Improving Transformer Optimization Through Better Initialization". Default: ``False``.
        no_layer_norm: if ``True`` all the layer norm layers in the module are removed.
            Should be used with t_fixup_init=True. Default: ``False``.
    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 t_fixup_init=False, no_layer_norm=False,
                 lattice_params: LatticeformerParams = LatticeformerParams(),
                 k_dim=0,
                 v_dim=0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        factory_kwargs = {}
        super(CrystalformerEncoderCUDA, self).__init__()
        assert lattice_params.domain in (
        "real", "reci", "RECI", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real")
        self.lattice_params = lattice_params

        self.layers = ModuleList([
            CrystalformerEncoderLayerCUDA(d_model, nhead, dim_feedforward, dropout,
                                          activation, layer_norm_eps, norm_first,
                                          no_layer_norm,
                                          params=lattice_params.getLayerParameters(i),
                                          k_dim=k_dim,
                                          v_dim=v_dim,
                                          **factory_kwargs)
            for i in range(num_encoder_layers)
        ])
        if t_fixup_init:
            for layer in self.layers:
                layer.fixup_initialization(num_encoder_layers)

        self.num_layers = num_encoder_layers
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else None

        self.d_model = d_model
        self.nhead = nhead
        v_head_dim = v_dim if v_dim > 0 else d_model // nhead
        k_head_dim = k_dim if k_dim > 0 else d_model // nhead
        compile_kernels(
            lattice_params.lattice_range, nhead,
            k_head_dim,
            lattice_params.value_pe_dist_real,
            lattice_params.value_pe_angle_real,
            v_head_dim,
            lattice_params.minimum_range,
            lattice_params.cos_abs,
            lattice_params.value_pe_dist_wscale,
            lattice_params.value_pe_angle_wscale,
            lattice_params.value_pe_dist_coef,
            lattice_params.value_pe_angle_coef,
            lattice_params.symm_break_noise
        )

    def forward(self, src: Tensor, pos: Tensor, batch: Tensor, trans: Tensor, sizes: Tensor, onehot: Tensor = None,
                this_epoch: int = None) -> Tensor:
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            batch: batch indices of the sequence (required).
            dist: the distances of lattice points to the encoder (required).
            edges: attention edges (required).
        Shape:
            - src: :math:`(S, E)`.
            - batch: `(S)`.
            - dist: :math:`(M, R)`, where M is the number of edges.
            - edges: :math:`(2, M)`.
            - output: :math:`(T, E)`.
            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        Examples:
            >>> output = transformer_model(src, src_mask=src_mask)
        """

        # for multi-gpu execution, adjust batch indices
        batch = batch - batch[0]
        P = self.lattice_params
        pos_ij, dist2_min, kr_base, rvecs, vcell, edges = compute_lattice_distances_for_cuda(
            pos, batch, trans, sizes, P.lattice_range,
            -1,  # P.scale_real*P.gauss_lb_real**(-0.5)*P.adaptive_cutoff_sigma,
            output_real=self.lattice_params.domain in (
            "real", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real"),
            output_reci=self.lattice_params.domain in (
            "reci", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real", "RECI"),
        )

        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        output = src

        for mod in self.layers:
            output = mod(output, batch, edges, pos_ij, dist2_min, trans, kr_base, rvecs, vcell, onehot, this_epoch)

        if self.norm is not None:
            output = self.norm(output)

        return output

class CrystalformerEncoderLayerCUDA(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 no_layer_norm=False,
                 params=LatticeformerParams(),
                 k_dim=0,
                 v_dim=0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        factory_kwargs = {}

        assert params.domain in ("real", "reci", "RECI", "multihead")
        self.domain = params.domain

        super(CrystalformerEncoderLayerCUDA, self).__init__()
        self.self_attn = CrystalformerMultiheadAttentionCUDA(
            d_model, nhead, dropout=dropout,
            kdim=k_dim,
            vdim=v_dim,
            params=params,
            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else Identity()
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else Identity()
        self.dropout1 = Dropout(dropout) if dropout > 0 else Identity()
        self.dropout2 = Dropout(dropout) if dropout > 0 else Identity()

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.add_zero_attn = False  # add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            constant_(self.linear1.bias, 0)
        if self.linear2.bias is not None:
            constant_(self.linear2.bias, 0)

    def fixup_initialization(self, num_layers):
        temp_state_dic = {}
        en_layers = num_layers

        for name, param in self.named_parameters():
            if name in ["linear1.weight",
                        "linear2.weight",
                        "self_attn.out_proj.weight",
                        "self_attn.v_proj_weight",
                        # "self_attn.pe_dist_proj",
                        # "self_attn.pe_wave_proj",
                        ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CrystalformerEncoderLayerCUDA, self).__setstate__(state)

    def forward(self, src: Tensor, batch: Tensor, edges: Tensor,
                pos_ij: Tensor = None, dist2_min: Tensor = None, trans: Tensor = None,
                kr_base: Tensor = None, rvecs: Tensor = None, vcell: Tensor = None,
                onehot: Tensor = None, this_epoch: int = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            batch: batch indices of the sequence (required).
            dist: the distances of lattice points to modulate attentions (required).
            edges: the attention edges (required).
        Shape:
            see the docs in Transformer class.
        """
        if self.domain in ("real", "multihead"):
            assert pos_ij is not None and trans is not None

        if self.domain in ("reci", "multihead", "RECI"):
            assert kr_base is not None and rvecs is not None and vcell is not None

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), batch, edges, pos_ij, dist2_min, trans, kr_base, rvecs, vcell, onehot,
                                   this_epoch)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, batch, edges, pos_ij, dist2_min, trans, kr_base, rvecs, vcell, onehot,
                                              this_epoch))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, batch: Tensor, edges: Tensor,
                  pos_ij: Tensor = None, dist2_min: Tensor = None, trans: Tensor = None,
                  kr_base: Tensor = None, rvecs: Tensor = None, vcell: Tensor = None,
                  onehot: Tensor = None, this_epoch: int = None) -> Tensor:
        x = self.self_attn(x, x, x, batch, batch, edges, pos_ij, dist2_min, trans, kr_base, rvecs, vcell, onehot,
                           need_weights=False, this_epoch=this_epoch)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class GradientScaler(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, onehot, sizes, scale):
    ctx.save_for_backward(onehot, sizes)
    ctx.scale = scale
    return x
  @staticmethod
  def backward(ctx, g):
    (onehot, sizes) = ctx.saved_tensors
    avr = avr_pool(onehot, sizes).mean(axis=0)
    w = onehot @ avr
    m = w>0
    w = w[m]
    w = (1/w)
    w /= w.mean()
    w.clamp_(max=ctx.scale)
    g[m] *= w[:,None]
    return g, None, None, None

class CrystalFramer(torch.nn.Module):
    """
    Latticeformer: str

    """

    def __init__(
        self,
        embedding_dim,
        num_layers=4,
        model_dim=128,
        ff_dim=512,
        t_fixup_init=True,
        pooling="max",
        pre_pooling_op="w+bn+relu",
        dropout=0.0,
        head_num=8,
        v_dim=0,
        k_dim=0,
        norm_type="bn",
        scale_grad=0.0,
        use_cuda_code=True,
        t_activation='relu',
        lattice_args=None,
        evidential="True"
    ):
        super().__init__()
        self.evidential=evidential
        embedding_dim = copy.deepcopy(embedding_dim)
        self.pooling = pooling
        self.scale_grad = scale_grad
        lattice_params = LatticeformerParams()
        lattice_params.parseFromArgs(lattice_args)

        print("lattice former params")
        print(lattice_params)

        self.ATOM_FEAT_DIM = 98
        self.input_embeddings = nn.Linear(self.ATOM_FEAT_DIM, model_dim, bias=False)
        emb_scale = model_dim ** (-0.5)
        if t_fixup_init:
            emb_scale *= (9 * num_layers) ** (-1 / 4)
        nn.init.normal_(self.input_embeddings.weight, mean=0, std=emb_scale)

        if use_cuda_code and not CUPY_AVAILABLE:
            print("Please install cupy and pytorch-pfn-extras to use the CUDA implementation.")
        Encoder = CrystalformerEncoderCUDA if use_cuda_code and CUPY_AVAILABLE else IndexedLatticeformerEncoder
        self.encoder = Encoder(
            model_dim,
            head_num,
            activation=t_activation,
            num_encoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            t_fixup_init=t_fixup_init,
            no_layer_norm=t_fixup_init,
            lattice_params=lattice_params,
            k_dim=k_dim,
            v_dim=v_dim)

        dim_pooled = model_dim
        if norm_type == "bn":
            norm_type = nn.BatchNorm1d
        elif norm_type == "ln":
            norm_type = nn.LayerNorm
        elif norm_type == "in":
            norm_type = nn.InstanceNorm1d
        elif norm_type in ["id", "no"]:
            norm_type = nn.Identity
        else:
            raise NotImplementedError(f"norm_type: {norm_type}")

        self.proj_before_pooling = lambda x: x
        if pre_pooling_op == "w+bn+relu":
            dim_pooled = embedding_dim.pop(0)
            self.proj_before_pooling = nn.Sequential(
                nn.Linear(model_dim, dim_pooled),
                norm_type(dim_pooled),
                nn.ReLU(True)
            )
        elif pre_pooling_op == "w+relu":
            dim_pooled = embedding_dim.pop(0)
            self.proj_before_pooling = nn.Sequential(
                nn.Linear(model_dim, dim_pooled),
                nn.ReLU(True)
            )
        elif pre_pooling_op == "relu":
            self.proj_before_pooling = nn.ReLU(True)
        elif pre_pooling_op == "no":
            pass
        else:
            raise NotImplementedError(f"pre_pooling_op: {pre_pooling_op}")

        if self.pooling == "max":
            self.pooling_layer = max_pool
        elif self.pooling == "avr":
            self.pooling_layer = avr_pool
        else:
            raise NotImplementedError(f"pooling: {self.pooling}")

        if self.evidential=="True":
            final_dim = 4
        else:
            final_dim = 1
        in_dim = [dim_pooled] + embedding_dim[:-1]
        out_dim = embedding_dim
        layers = []
        for di, do in zip(in_dim, out_dim):
            layers.append(nn.Linear(di, do))
            layers.append(norm_type(do))
            layers.append(nn.ReLU(True))
        if self.evidential=="True":
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(out_dim[-1], final_dim))
        self.mlp = nn.Sequential(*layers)
        self.out_act = nn.Softplus()

    def forward(self, data):
        min_val = 1e-6
        x = data.x
        pos = data.pos
        batch = data.batch
        trans = data.trans_vec
        sizes = data.sizes
        onehot_x = x

        x = self.input_embeddings(x)
        if self.scale_grad > 0:
            x = GradientScaler().apply(x, onehot_x, sizes, self.scale_grad)

        x = self.encoder(x, pos, batch, trans, sizes)
        # x: (total_point_num, d_model)

        x = self.proj_before_pooling(x)
        if self.pooling.startswith("pma"):
            x = self.pooling_layer(x, batch, sizes.shape[0])
        else:
            x = self.pooling_layer(x, batch, sizes)

        out = self.mlp(x)

        if self.evidential=="True":
            if out.shape[0] == 4:
                out = torch.unsqueeze(out, 0)
            out = out.view(out.shape[0], -1, 4)
            mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(out, 1, dim=-1)]
            return mu, self.out_act(logv)+ min_val, self.out_act(logalpha)+ min_val + 1, self.out_act(logbeta)+ min_val
        else:
            return torch.squeeze(out)

