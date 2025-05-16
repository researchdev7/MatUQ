import sys
import os
import csv
import json
import math
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Subset
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core.structure import Structure
from e3nn import o3
from e3nn.math import perm
from e3nn.util.jit import compile_mode
from e3nn.o3 import Irreps
from e3nn.math import normalize2mom

class AtomInitializer(object):
    """
    Base class for initializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_features(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self):
        # 92 dimensions
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------

        distances: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


# defining periodic graph within cutoff.
def get_radius_graph_knn(structure, cutoff, max_neighbors):
    MNN = MinimumDistanceNN(cutoff=cutoff, get_all_sites=True)
    edge_src, edge_dest, edge_vec, distance = [], [], [], []

    for i in range(len(structure.sites)):
        start = i
        center_site = np.array(structure[i].coords)
        mdnn = MNN.get_nn_info(structure, i)
        atom_radius_i = []
        for elem_i, occu_i in structure[i].species.items():
            atom_radius_i.append(elem_i.atomic_radius)
        center_max_radius = max(atom_radius_i)

        for atom in mdnn:
            end = atom['site_index']
            end_coords = np.array(atom['site'].coords, dtype=object)
            atom_radius_j = []
            for elem_j, occu_j in atom['site'].species.items():
                atom_radius_j.append(elem_j.atomic_radius)
            neigh_max_radius = max(atom_radius_j)
            try:
                radius = center_max_radius + neigh_max_radius
            except:
                radius = 0
            if np.array(atom['site'], dtype=object)[1] < radius:
                continue
            edge_src += [start]
            edge_dest += [end]
            edge_vec_t = np.array(center_site) - np.array(end_coords)
            edge_vec.append(edge_vec_t)
            distance.append(np.array(atom['site'], dtype=object)[1])

    edge_src, edge_dest, edge_vec, edge_distances = np.array(edge_src), np.array(edge_dest), np.array(
        edge_vec), np.array(distance)

    max_neigh_index = np.array([])
    ## KNN methods
    for i in range(len(structure.sites)):
        idx_i = (edge_src == i).nonzero()[0]
        distance_sorted = np.sort(edge_distances[idx_i])
        # To include self edge, not using max_neighbors -1 ;
        if len(distance_sorted) != 0:
            try:
                max_dist = distance_sorted[max_neighbors - 1]
            except:
                max_dist = distance_sorted[-1]
            max_dist_index = np.where(edge_distances[idx_i] <= max_dist + 0.001)
            max_dist_index = np.array(max_dist_index).flatten()

            max_neigh_index_t = [idx_i[i] for i in max_dist_index]
            max_neigh_index_t = np.array(max_neigh_index_t)
            max_neigh_index = np.append(max_neigh_index, max_neigh_index_t)

    max_neigh_index = max_neigh_index.flatten().astype(int)
    max_neigh_index = [max_neigh_index[i] for i in range(len(max_neigh_index))]

    edge_src, edge_dest, edge_vec, distances = edge_src[max_neigh_index], edge_dest[max_neigh_index], edge_vec[
        max_neigh_index], edge_distances[max_neigh_index]

    return edge_src, edge_dest, edge_vec, distances


def get_sodnet_train_val_test_loader(dataset, train_indexs=None,
                              val_indexs=None, test_indexs=None,
                              batch_size=256, return_test=True,
                              num_workers=0, pin_memory=False):

    total_size = len(dataset)
    train_indice = []
    val_indice = []
    test_indice = []
    for i in range(total_size):
        cif_id = dataset[i].name
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
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

################################################################################
# Pytorch datasets
################################################################################

##Fetch dataset; processes the raw data if specified
def SODNetData(data_path, task):

    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    if os.path.exists(os.path.join(data_path, task, "sodnet_data.pt")) == True:
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
        file_names = ["sodnet_data.pt"]
        return file_names

def process_data(data_path, task):
    ##Begin processing data
    print("Processing data to: " + os.path.join(data_path, task))
    assert os.path.exists(data_path), "Data path not found in " + data_path

    ##Load dictionary
    dictionary_file_path = os.path.join(
        data_path, "sodnet_atom_embedding.json"
    )
    if os.path.exists(dictionary_file_path) == False:
        print("Atom dictionary not found, exiting program...")
        sys.exit()
    else:
        print("Loading atom dictionary from file.")
        ari = AtomCustomJSONInitializer(dictionary_file_path)

    ##Load targets
    target_property_file = os.path.join(data_path, task, "targets.csv")
    assert os.path.exists(target_property_file), (
            "targets not found in " + target_property_file
    )
    with open(target_property_file) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader]

    ##Process structure files and create structure graphs
    r_cut = 8
    max_neighbors = 32
    data_list = []
    for index in range(0, len(target_data)):

        structure_id = target_data[index][0]
        crystal = Structure.from_file(os.path.join(
            data_path, task, structure_id + ".cif"))
        crystal = crystal.get_reduced_structure()
        crystal = crystal.get_primitive_structure()

        num_nodes = len(crystal.sites)

        atom_features = []
        occu_crystal = []
        for i in range(len(crystal.sites)):
            emb = 0
            total = 0
            for ele, occup in crystal[i].species.items():
                num = ele.number
                feature = np.vstack(ari.get_atom_features(num))
                emb += feature * occup
                total += occup
            atom_features.append(emb)
            occu_crystal.append(total)

        x = torch.tensor(atom_features).reshape((int(num_nodes), -1))
        edge_src, edge_dst, edge_vec, distances = get_radius_graph_knn(crystal, r_cut, max_neighbors)

        edge_occu = []
        for src, dst in zip(edge_src, edge_dst):
            occu = occu_crystal[src] * occu_crystal[dst]
            edge_occu.append(occu)

        distances = np.array(distances)
        name = structure_id

        # build atom pairs within cutoff
        edge_num = len(edge_src)
        edge_num = torch.tensor(edge_num, dtype=torch.long)
        edge_src = torch.tensor(edge_src, dtype=torch.long)
        edge_dst = torch.tensor(edge_dst, dtype=torch.long)
        edge_occu = torch.tensor(edge_occu, dtype=torch.float)

        edge_vec = torch.tensor(edge_vec.astype(float), dtype=torch.float)
        edge_attr = torch.tensor(distances, dtype=torch.float)
        target = target_data[index][1]
        y = torch.Tensor([float(target)])
        data = Data(x=x, edge_occu=edge_occu, edge_src=edge_src, edge_dst=edge_dst,
                    edge_attr=edge_attr, y=y, name=name, index=index,
                    edge_vec=edge_vec, edge_num=edge_num)
        data_list.append(data)

    if os.path.isdir(os.path.join(data_path, task)) == False:
        os.mkdir(os.path.join(data_path, task))

    ##Save processed dataset to file
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(data_path, task, "sodnet_data.pt"))

_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 100
_AVG_NUM_NODES = 29.891087392943284
_AVG_DEGREE = 34.29242574467496

class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)

def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


@compile_mode('script')
class Gate(torch.nn.Module):
    '''
        1. Use `narrow` to split tensor.
        2. Use `Activation` in this file.
    '''

    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}")
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates")
        # assert len(irreps_scalars) == 1
        # assert len(irreps_gates) == 1

        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated
        self._irreps_in = (irreps_scalars + irreps_gates + irreps_gated).simplify()

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self._irreps_out = irreps_scalars + irreps_gated

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        scalars_dim = self.irreps_scalars.dim
        gates_dim = self.irreps_gates.dim
        input_dim = self.irreps_in.dim
        scalars = features.narrow(-1, 0, scalars_dim)
        gates = features.narrow(-1, scalars_dim, gates_dim)
        gated = features.narrow(-1, (scalars_dim + gates_dim),
                                (input_dim - scalars_dim - gates_dim))

        scalars = self.act_scalars(scalars)
        if gates.shape[-1]:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            features = torch.cat([scalars, gated], dim=-1)
        else:
            features = scalars
        return features

    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in

    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out
class TensorProductRescale(torch.nn.Module):
    def __init__(self,
                 irreps_in1, irreps_in2, irreps_out,
                 instructions,
                 bias=True, rescale=True,
                 internal_weights=None, shared_weights=None,
                 normalization=None):

        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias

        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        self.tp = o3.TensorProduct(irreps_in1=self.irreps_in1,
                                   irreps_in2=self.irreps_in2, irreps_out=self.irreps_out,
                                   instructions=instructions, normalization=normalization,
                                   internal_weights=internal_weights, shared_weights=shared_weights,
                                   path_normalization='none')

        self.init_rescale_bias()

    def calculate_fan_in(self, ins):
        return {
            'uvw': (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            'uvu': self.irreps_in2[ins.i_in2].mul,
            'uvv': self.irreps_in1[ins.i_in1].mul,
            'uuw': self.irreps_in1[ins.i_in1].mul,
            'uuu': 1,
            'uvuv': 1,
            'uvu<v': 1,
            'u<vw': self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
        }[ins.connection_mode]

    def init_rescale_bias(self) -> None:

        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_parity = [irrep_str[-1] for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(self.irreps_bias).split('+')]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if self.irreps_bias_orders[slice_idx] == 0 and self.irreps_bias_parity[slice_idx] == 'e':
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype))
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                            fan_in if slice_idx in slices_fan_in.keys() else fan_in)
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    # else:
                    #    sqrt_k = 1.
                    #
                    # if self.rescale:
                    # weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    # self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            # for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)

    def forward_tp_rescale_bias(self, x, y, weight=None):

        out = self.tp(x, y, weight)

        # if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for (_, slice, bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                # out[:, slice] += bias
                out.narrow(1, slice.start, slice.stop - slice.start).add_(bias)
        return out

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out


def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output,
                           internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined.
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
                    for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
                              irreps_output, instructions,
                              internal_weights=internal_weights,
                              shared_weights=internal_weights,
                              bias=bias, rescale=_RESCALE)
    return tp


class SeparableFCTP(torch.nn.Module):
    '''
        Use separable FCTP for spatial convolution.
    '''

    def __init__(self, irreps_node_input, irreps_edge_attr, irreps_node_output,
                 fc_neurons, use_activation=False, internal_weights=False):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.dtp = DepthwiseTensorProduct(self.irreps_node_input, self.irreps_edge_attr,
                                          self.irreps_node_output, bias=False, internal_weights=internal_weights)

        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k

        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)

        self.norm = None

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate

    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by
            self.dtp_rad(`edge_scalars`).
        '''
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out


@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out

    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)

@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''

    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out

    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)

class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(self,
                 irreps_in1, irreps_in2, irreps_out,
                 bias=True, rescale=True,
                 internal_weights=None, shared_weights=None,
                 normalization=None):
        instructions = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out,
                         instructions=instructions,
                         bias=bias, rescale=rescale,
                         internal_weights=internal_weights, shared_weights=shared_weights,
                         normalization=normalization)

def irreps2gate(irreps):
    irreps_scalars = []
    irreps_gated = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            irreps_scalars.append((mul, ir))
        else:
            irreps_gated.append((mul, ir))
    irreps_scalars = o3.Irreps(irreps_scalars).simplify()
    irreps_gated = o3.Irreps(irreps_gated).simplify()
    if irreps_gated.dim > 0:
        ir = '0e'
    else:
        ir = None
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
    return irreps_scalars, irreps_gates, irreps_gated
class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):

    def __init__(self, irreps_in1, irreps_in2, irreps_out,
                 bias=True, rescale=True,
                 internal_weights=None, shared_weights=None,
                 normalization=None):

        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
                         bias=bias, rescale=rescale,
                         internal_weights=internal_weights, shared_weights=shared_weights,
                         normalization=normalization)
        self.gate = gate

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out

class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True):
        super().__init__(irreps_in, o3.Irreps('1x0e'), irreps_out,
                         bias=bias, rescale=rescale, internal_weights=True,
                         shared_weights=True, normalization=None)

    def forward(self, x):
        y = torch.ones_like(x[:, 0:1])
        out = self.forward_tp_rescale_bias(x, y)
        return out

class NodeEmbeddingNetwork(torch.nn.Module):

    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)),
                                      self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)

    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
        '''
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)

        return node_embedding, node_attr, node_atom_onehot


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0
        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, dist, node_atom=None, edge_src=None, edge_dst=None):
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        x = gaussian(x, mean, std)
        return x


def sort_irreps_even_first(irreps):
    Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
    out = [(ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(irreps)]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    p = perm.inverse(inv)
    irreps = o3.Irreps([(mul, (l, -p)) for l, p, _, mul in out])
    return Ret(irreps, p, inv)

class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue
            if (i == len(ch_list) - 1) and use_offset:
                use_biases = False
            else:
                use_biases = True
            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_biases))
            input_channels = ch_list[i]

            if i == len(ch_list) - 1:
                break

            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))
            # modules.append(nn.ReLU())
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[torch.nn.functional.silu]))
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[ShiftedSoftplus()]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(ch_list[-1]))
            fan_in = ch_list[-2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.offset, -bound, bound)

    def forward(self, f_in):
        f_out = self.net(f_in)
        if self.offset is not None:
            f_out = f_out + self.offset.reshape(1, -1)
        return f_out
class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0

    def forward(self, x, index, **kwargs):
        out = scatter(x, index, reduce='mean', **kwargs)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out

    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)
class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        super().__init__()
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding,
                            bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding,
                                         irreps_edge_attr, irreps_node_embedding,
                                         internal_weights=False, bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)

    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0,
                                           dim_size=node_features.shape[0])
        return node_features

class EquivariantLayerNormV2(nn.Module):

    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input, **kwargs):
        # batch, *size, dim = node_input.shape  # TODO: deal with batch
        # node_input = node_input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # node_input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            # field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = torch.mean(field, dim=1, keepdim=True)  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output

class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(irreps,
                                               o3.Irreps('{}x0e'.format(self.num_irreps)))

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out


@compile_mode('trace')
class Activation(torch.nn.Module):
    '''
        Directly apply activation when irreps is type-0.
    '''

    def __init__(self, irreps_in, acts):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError("Activation: cannot apply an activation function to a non-scalar input.")

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd.")
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(acts)
        assert len(self.irreps_in) == len(self.acts)

    # def __repr__(self):
    #    acts = "".join(["x" if a is not None else " " for a in self.acts])
    #    return f"{self.__class__.__name__} [{self.acts}] ({self.irreps_in} -> {self.irreps_out})"
    def extra_repr(self):
        output_str = super(Activation, self).extra_repr()
        output_str = output_str + '{} -> {}, '.format(self.irreps_in, self.irreps_out)
        return output_str

    def forward(self, features, dim=-1):
        # directly apply activation without narrow
        if len(self.acts) == 1:
            return self.acts[0](features)

        output = []
        index = 0
        for (mul, ir), act in zip(self.irreps_in, self.acts):
            if act is not None:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * ir.dim))
            index += mul * ir.dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)

@compile_mode('script')
class GraphAttention(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''

    def __init__(self,
                 irreps_node_input, irreps_node_attr,
                 irreps_edge_attr, irreps_node_output,
                 fc_neurons,
                 irreps_head, num_heads, irreps_pre_attn=None,
                 rescale_degree=False, nonlinear_message=False,
                 alpha_drop=0.1, proj_drop=0.1):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)

        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads)  # irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify()
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha))  # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()

        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(self.irreps_pre_attn,
                                         self.irreps_edge_attr, self.irreps_pre_attn, fc_neurons,
                                         use_activation=True, internal_weights=False)
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(self.irreps_pre_attn,
                                           self.irreps_edge_attr, irreps_attn_heads, fc_neurons=None,
                                           use_activation=False, internal_weights=True)
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)),
                                                 num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(self.irreps_pre_attn,
                                     self.irreps_edge_attr, irreps_attn_all, fc_neurons,
                                     use_activation=False)
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(),
                num_heads)

        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)),
                                    [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)

        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot)  # Following GATv2

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input,
                                                drop_prob=proj_drop)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars,
                batch, **kwargs):

        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]

        if self.nonlinear_message:
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))

        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)

        if self.rescale_degree:
            degree = torch_geometric.utils.degree(edge_dst,
                                                  num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree

        node_output = self.proj(attn)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class GraphDropPath(nn.Module):
    '''
        Consider batch for graph data when dropping paths.
    '''

    def __init__(self, drop_prob=None):
        super(GraphDropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out

    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)

@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    '''
        Use two (FCTP + Gate)
    '''

    def __init__(self,
                 irreps_node_input, irreps_node_attr,
                 irreps_node_output, irreps_mlp_mid=None,
                 proj_drop=0.1):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid,
            bias=True, rescale=_RESCALE)
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output,
            bias=True, rescale=_RESCALE)

        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_output,
                                                drop_prob=proj_drop)

    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output


@compile_mode('script')
class TransBlock(torch.nn.Module):
    '''
        1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
        2. Use pre-norm architecture
    '''

    def __init__(self,
                 irreps_node_input, irreps_node_attr,
                 irreps_edge_attr, irreps_node_output,
                 fc_neurons,
                 irreps_head, num_heads, irreps_pre_attn=None,
                 rescale_degree=False, nonlinear_message=False,
                 alpha_drop=0.1, proj_drop=0.1,
                 drop_path_rate=0.0,
                 irreps_mlp_mid=None,
                 ):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input

        self.resweight = torch.nn.Parameter(torch.Tensor([0]))
        self.ga = GraphAttention(irreps_node_input=self.irreps_node_input,
                                 irreps_node_attr=self.irreps_node_attr,
                                 irreps_edge_attr=self.irreps_edge_attr,
                                 irreps_node_output=self.irreps_node_input,
                                 fc_neurons=fc_neurons,
                                 irreps_head=self.irreps_head,
                                 num_heads=self.num_heads,
                                 irreps_pre_attn=self.irreps_pre_attn,
                                 rescale_degree=self.rescale_degree,
                                 nonlinear_message=self.nonlinear_message,
                                 alpha_drop=alpha_drop,
                                 proj_drop=proj_drop)

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None

        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input, self.irreps_node_attr,
                self.irreps_node_output,
                bias=True, rescale=_RESCALE)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars,
                batch, **kwargs):

        node_output = node_input
        node_features = node_input
        node_features = self.ga(node_input=node_features,
                                node_attr=node_attr,
                                edge_src=edge_src, edge_dst=edge_dst,
                                edge_attr=edge_attr, edge_scalars=edge_scalars,
                                batch=batch)
        node_features = node_features * self.resweight

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        node_features = node_output
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)

        node_features = node_features * self.resweight
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        return node_output


class SODNet(torch.nn.Module):
    def __init__(self,
                 irreps_in='100x0e',
                 irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                 irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                 max_radius=5.0,
                 number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64],
                 irreps_feature='512x0e',
                 irreps_head='16x0e+8x1e+4x2e', num_heads=8, irreps_pre_attn=None,
                 rescale_degree=False, nonlinear_message=True,
                 irreps_mlp_mid='384x0e+192x1e+96x2e',
                 alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
                 drop_path_rate=0.0,
                 mean=None, std=None, scale=None, atomref=None,
                 evidential="False",
                 classification=False,
                 ):
        super().__init__()

        self.classification = classification
        self.evidential = evidential
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding,
                                                         self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)

        self.blocks = torch.nn.ModuleList()
        self.build_blocks()

        self.norm = EquivariantLayerNormV2(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]))

        self.dropout = nn.Dropout(p=0.1)
        self.out_act = nn.Softplus()
        if self.classification:
            self.out = LinearRS(self.irreps_feature, o3.Irreps('2x0e'), rescale=_RESCALE)
            self.softmax = nn.Sigmoid()
            # self.softmax = nn.LogSoftmax(dim=1)
        elif self.evidential == "True":
            self.out = LinearRS(self.irreps_feature, o3.Irreps('4x0e'), rescale=_RESCALE)
        else:
            self.out = LinearRS(self.irreps_feature, o3.Irreps('1x0e'), rescale=_RESCALE)

        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        self.apply(self._init_weights)

    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding,
                             irreps_node_attr=self.irreps_node_attr,
                             irreps_edge_attr=self.irreps_edge_attr,
                             irreps_node_output=irreps_block_output,
                             fc_neurons=self.fc_neurons,
                             irreps_head=self.irreps_head,
                             num_heads=self.num_heads,
                             irreps_pre_attn=self.irreps_pre_attn,
                             rescale_degree=self.rescale_degree,
                             nonlinear_message=self.nonlinear_message,
                             alpha_drop=self.alpha_drop,
                             proj_drop=self.proj_drop,
                             drop_path_rate=self.drop_path_rate,
                             irreps_mlp_mid=self.irreps_mlp_mid,
                             )
            self.blocks.append(blk)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear)
                    or isinstance(module, torch.nn.LayerNorm)
                    or isinstance(module, EquivariantLayerNormV2)
                    or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    def forward(self, f_in, edge_occu, edge_src, edge_dst, edge_vec, edge_attr, edge_num,
                batch, **kwargs):
        min_val = 1e-6
        f_in = f_in.to(torch.float32)
        batch_counts = torch.cat((
            torch.zeros(1, dtype=torch.long, device=f_in.device),
            torch.cumsum(torch.bincount(batch), dim=0)[:-1]
        ))

        counts = torch.cat([batch_counts[i].repeat(edge_num[i]) for i in range(len(edge_num))])
        edge_src.add_(counts)
        edge_dst.add_(counts)

        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
                                         x=edge_vec, normalize=True, normalization='component')

        atom_embedding = f_in
        edge_length = edge_attr
        edge_occu = edge_occu.unsqueeze(-1)
        edge_occu = edge_occu.expand(-1, self.number_of_basis)
        edge_length_embedding = torch.mul(self.rbf(edge_length), edge_occu)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh,
                                                    edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr,
                                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh,
                                edge_scalars=edge_length_embedding,
                                batch=batch)

        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        outputs = self.head(node_features)
        outputs = self.dropout(outputs)
        outputs = self.out(outputs)
        outputs = self.scale_scatter(outputs, batch, dim=0)

        if self.scale is not None:
            outputs = self.scale * outputs

        if self.classification:
            # out = torch.max(out,dim=1)
            outputs = self.softmax(outputs)
        if self.evidential=="True":
            if outputs.shape[0] == 4:
                outputs = torch.unsqueeze(outputs, 0)
            outputs = outputs.view(outputs.shape[0], -1, 4)
            mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(outputs, 1, dim=-1)]
            return mu, self.out_act(logv)+ min_val, self.out_act(logalpha)+ min_val + 1, self.out_act(logbeta)+ min_val
        else:
            return torch.squeeze(outputs)
