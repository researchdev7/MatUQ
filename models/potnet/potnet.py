import os
import sys
import csv
import math
import numpy as np
from typing import Optional, Union
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
from jarvis.core.specie import chem_data, get_node_attributes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models.schnet import ShiftedSoftplus
from torch_geometric.nn import Linear, MessagePassing, global_mean_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_sparse import SparseTensor
#from models.potnet import algorithm
import periodictable

group_id = {
            "H": 0,
            "He": 1,
            "Li": 2,
            "Be": 3,
            "B": 4,
            "C": 0,
            "N": 0,
            "O": 0,
            "F": 5,
            "Ne": 1,
            "Na": 2,
            "Mg": 3,
            "Al": 6,
            "Si": 4,
            "P": 0,
            "S": 0,
            "Cl": 5,
            "Ar": 1,
            "K": 2,
            "Ca": 3,
            "Sc": 7,
            "Ti": 7,
            "V": 7,
            "Cr": 7,
            "Mn": 7,
            "Fe": 7,
            "Co": 7,
            "Ni": 7,
            "Cu": 7,
            "Zn": 7,
            "Ga": 6,
            "Ge": 4,
            "As": 4,
            "Se": 0,
            "Br": 5,
            "Kr": 1,
            "Rb": 2,
            "Sr": 3,
            "Y": 7,
            "Zr": 7,
            "Nb": 7,
            "Mo": 7,
            "Tc": 7,
            "Ru": 7,
            "Rh": 7,
            "Pd": 7,
            "Ag": 7,
            "Cd": 7,
            "In": 6,
            "Sn": 6,
            "Sb": 4,
            "Te": 4,
            "I": 5,
            "Xe": 1,
            "Cs": 2,
            "Ba": 3,
            "La": 8,
            "Ce": 8,
            "Pr": 8,
            "Nd": 8,
            "Pm": 8,
            "Sm": 8,
            "Eu": 8,
            "Gd": 8,
            "Tb": 8,
            "Dy": 8,
            "Ho": 8,
            "Er": 8,
            "Tm": 8,
            "Yb": 8,
            "Lu": 8,
            "Hf": 7,
            "Ta": 7,
            "W": 7,
            "Re": 7,
            "Os": 7,
            "Ir": 7,
            "Pt": 7,
            "Au": 7,
            "Hg": 7,
            "Tl": 6,
            "Pb": 6,
            "Bi": 6,
            "Po": 4,
            "At": 5,
            "Rn": 1,
            "Fr": 2,
            "Ra": 3,
            "Ac": 9,
            "Th": 9,
            "Pa": 9,
            "U": 9,
            "Np": 9,
            "Pu": 9,
            "Am": 9,
            "Cm": 9,
            "Bk": 9,
            "Cf": 9,
            "Es": 9,
            "Fm": 9,
            "Md": 9,
            "No": 9,
            "Lr": 9,
            "Rf": 7,
            "Db": 7,
            "Sg": 7,
            "Bh": 7,
            "Hs": 7
        }
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
        file_names = ["potnet_data.pt"]
        return file_names

def process_data(data_path, task, processing_args):
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

    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        max_z = max(v["Z"] for v in chem_data.values())

        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features
    ##Process structure files and create structure graphs
    data_list = []
    features = _get_attribute_lookup(processing_args["atom_features"])
    for index in range(0, len(target_data)):

        data = Data()
        structure_id = target_data[index][0]
        data.structure_id = [structure_id]
        crystal = Atoms.from_cif(os.path.join(data_path, task, structure_id + '.cif'))
        crystal = crystal.to_dict()
        structure = (
            Atoms.from_dict(crystal) if isinstance(crystal, dict) else crystal
        )

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(structure.elements):
            feat = list(get_node_attributes(s, atom_features="atomic_number"))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )

        u = torch.arange(0, node_features.size(0), 1).unsqueeze(1).repeat((1, node_features.size(0))).flatten().long()
        v = torch.arange(0, node_features.size(0), 1).unsqueeze(0).repeat((node_features.size(0), 1)).flatten().long()

        edge_index = torch.stack([u, v])

        lattice_mat = structure.lattice_mat.astype(dtype=np.double)

        vecs = structure.cart_coords[u.flatten().numpy().astype(int)] - structure.cart_coords[
            v.flatten().numpy().astype(int)]

        inf_edge_attr = torch.FloatTensor(np.stack([getattr(algorithm, func)(vecs, lattice_mat, param=param, R=processing_args["R"])
                                     for func, param in zip(processing_args["infinite_funcs"], processing_args["infinite_params"])], 1))
        edges = nearest_neighbor_edges(atoms=structure, cutoff=4, max_neighbors=16)
        u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)

        data.atom_numbers = node_features
        data.edge_attr = r.norm(dim=-1)
        data.edge_index = torch.stack([u, v])
        data.inf_edge_index = edge_index
        data.inf_edge_attr = inf_edge_attr

        group_feats = []
        for atom in node_features:
            group_feats.append(group_id[periodictable.elements[int(atom)].symbol])
        group_feats = torch.tensor(np.array(group_feats)).type(torch.LongTensor)
        identity_matrix = torch.eye(10)
        g_feats = identity_matrix[group_feats]
        if len(list(g_feats.size())) == 1:
            g_feats = g_feats.unsqueeze(0)

        f = torch.tensor(features[data.atom_numbers.long().squeeze(1)]).type(torch.FloatTensor)
        if len(data.atom_numbers) == 1:
            f = f.unsqueeze(0)

        data.x = f
        data.g_feats = g_feats
        target = target_data[index][1]
        y = torch.Tensor([float(target)])
        data.y = y
        data_list.append(data)

    if os.path.isdir(os.path.join(data_path, task)) == False:
        os.mkdir(os.path.join(data_path, task))

    ##Save processed dataset to file
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(data_path, task, "potnet_data.pt"))

def load_potnet_data(data_path, task, config=None):

    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    if os.path.exists(os.path.join(data_path, task, "potnet_data.pt")) == True:
        dataset = StructureDataset(
            data_path,
            task,
        )
    else:
        process_data(data_path, task, config["Models"])
        dataset = StructureDataset(
            data_path,
            task,
        )
    return dataset

def get_potnet_train_val_test_loader(dataset, train_indexs=None,
                              val_indexs=None, test_indexs=None,
                              batch_size=256, return_test=True,
                              num_workers=0, pin_memory=False):

    total_size = len(dataset)
    train_indice = []
    val_indice = []
    test_indice = []
    for i in range(total_size):
        cif_id = dataset[i].structure_id[0]
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
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if return_test:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
            type: str = "gaussian"
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(vmin, vmax, bins)
        )
        self.type = type

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        base = self.gamma * (distance.unsqueeze(-1) - self.centers)
        if self.type == 'gaussian':
            return (-base ** 2).exp()
        elif self.type == 'quadratic':
            return base ** 2
        elif self.type == 'linear':
            return base
        elif self.type == 'inverse_quadratic':
            return 1.0 / (1.0 + base ** 2)
        elif self.type == 'multiquadric':
            return (1.0 + base ** 2).sqrt()
        elif self.type == 'inverse_multiquadric':
            return 1.0 / (1.0 + base ** 2).sqrt()
        elif self.type == 'spline':
            return base ** 2 * (base + 1.0).log()
        elif self.type == 'poisson_one':
            return (base - 1.0) * (-base).exp()
        elif self.type == 'poisson_two':
            return (base - 2.0) / 2.0 * base * (-base).exp()
        elif self.type == 'matern32':
            return (1.0 + 3 ** 0.5 * base) * (-3 ** 0.5 * base).exp()
        elif self.type == 'matern52':
            return (1.0 + 5 ** 0.5 * base + 5 / 3 * base ** 2) * (-5 ** 0.5 * base).exp()
        else:
            raise Exception("No Implemented Radial Basis Method")

class TransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            hidden_dim: int,
            edge_dim: int,
            bias: bool = True,
            root_weight: bool = True,
            aggr: str = 'mean',
            **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.hidden_dim = hidden_dim
        self.root_weight = root_weight
        self.edge_dim = edge_dim
        self._alpha = None

        self.lin_edge = Linear(edge_dim, hidden_dim)

        self.lin_concate = Linear(hidden_dim, hidden_dim)

        self.lin_skip = Linear(hidden_dim, hidden_dim, bias=bias)

        self.lin_msg_update = Linear(hidden_dim, hidden_dim)
        self.msg_layer = Linear(hidden_dim, hidden_dim)
        self.msg_ln = nn.LayerNorm(hidden_dim)

        self.alpha_ln = nn.LayerNorm(hidden_dim)

        self.bn = BatchNorm(hidden_dim)
        self.skip_bn = BatchNorm(hidden_dim)
        self.act = ShiftedSoftplus()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: Tensor, return_attention_weights=None):
        out = self.propagate(edge_index, query=x, key=x, value=x, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        out = self.act(self.bn(out))

        if self.root_weight:
            out = out + x

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = self.lin_edge(edge_attr)
        alpha = (query_i * key_j) / math.sqrt(self.hidden_dim) + edge_attr
        alpha = F.sigmoid(self.act(self.alpha_ln(alpha)))
        self._alpha = alpha

        out = self.lin_msg_update(value_j) * alpha
        out = F.relu(self.msg_ln(self.msg_layer(out)))
        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(, '
                f'{self.hidden_dim})')


class PotNetConv(MessagePassing):

    def __init__(self, fc_features):
        super(PotNetConv, self).__init__(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )

        return F.relu(x + self.bn(out))

    def message(self, x_i, x_j, edge_attr, index):
        score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1))))
        return score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))


class PotNet(nn.Module):

    def __init__(
        self,
        conv_layers: int = 3,
        atom_input_features: int = 92,
        inf_edge_features: int = 64,
        fc_features: int = 256,
        output_dim: int = 256,
        output_features: int = 1,
        evidential: str = "False",
        classification: bool = False,
        rbf_min = -4.0,
        rbf_max = 4.0,
        potentials = [],
        euclidean = False,
        charge_map = False,
        transformer = False,
    ):
        super().__init__()
        self.classification = classification
        self.evidential = evidential
        self.euclidean = euclidean
        self.potentials = potentials
        self.charge_map = charge_map
        self.transformer = transformer
        self.convs = conv_layers
        if not charge_map:
            self.atom_embedding = nn.Linear(
                atom_input_features, fc_features
            )
        else:
            self.atom_embedding = nn.Linear(
                atom_input_features + 10, fc_features
            )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=rbf_min,
                vmax=rbf_max,
                bins=fc_features,
            ),
            nn.Linear(fc_features, fc_features),
            nn.SiLU(),
        )

        if not euclidean:
            self.inf_edge_embedding = RBFExpansion(
                vmin=rbf_min,
                vmax=rbf_max,
                bins=inf_edge_features,
                type='multiquadric'
            )

            self.infinite_linear = nn.Linear(inf_edge_features, fc_features)

            self.infinite_bn = nn.BatchNorm1d(fc_features)

        self.conv_layers = nn.ModuleList(
            [
                PotNetConv(fc_features)
                for _ in range(conv_layers)
            ]
        )

        if not euclidean and transformer:
            self.transformer_conv_layers = nn.ModuleList(
                [
                    TransformerConv(fc_features, fc_features)
                    for _ in range(conv_layers)
                ]
            )

        self.fc = nn.Sequential(
            nn.Linear(fc_features, fc_features), ShiftedSoftplus()
        )

        if self.classification:
            self.fc_out = nn.Linear(fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        elif self.evidential == "True":
            self.fc_out = nn.Linear(output_dim, 4)
            self.out_act = nn.Softplus()
        else:
            self.fc_out = nn.Linear(output_dim, output_features)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, data):
        """CGCNN function mapping graph to outputs."""
        min_val = 1e-6
        # fixed edge features: RBF-expanded bondlengths
        edge_index = data.edge_index
        if self.euclidean:
            edge_features = self.edge_embedding(data.edge_attr)
        else:
            edge_features = self.edge_embedding(-0.75 / data.edge_attr)

        if not self.euclidean:
            inf_edge_index = data.inf_edge_index
            inf_feat = sum([data.inf_edge_attr[:, i] * pot for i, pot in enumerate(self.potentials)])
            inf_edge_features = self.inf_edge_embedding(inf_feat)
            inf_edge_features = self.infinite_bn(F.softplus(self.infinite_linear(inf_edge_features)))

        # initial node features: atom feature network...
        if self.charge_map:
            node_features = self.atom_embedding(torch.cat([data.x, data.g_feats], -1))
        else:
            node_features = self.atom_embedding(data.x)

        if not self.euclidean and not self.transformer:
            edge_index = torch.cat([data.edge_index, inf_edge_index], 1)
            edge_features = torch.cat([edge_features, inf_edge_features], 0)

        for i in range(self.convs):
            if not self.euclidean and self.transformer:
                local_node_features = self.conv_layers[i](node_features, edge_index, edge_features)
                inf_node_features = self.transformer_conv_layers[i](node_features, inf_edge_index, inf_edge_features)
                node_features = local_node_features + inf_node_features
            else:
                node_features = self.conv_layers[i](node_features, edge_index, edge_features)

        features = global_mean_pool(node_features, data.batch)
        features = self.fc(features)
        features = self.dropout(features)
        out = self.fc_out(features)

        if self.classification:
            out = self.softmax(out)
        if self.evidential=="True":
            if out.shape[0] == 4:
                out = torch.unsqueeze(out, 0)
            out = out.view(out.shape[0], -1, 4)
            mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(out, 1, dim=-1)]
            return mu, self.out_act(logv)+ min_val, self.out_act(logalpha)+ min_val + 1, self.out_act(logbeta)+ min_val
        else:
            return torch.squeeze(out)
