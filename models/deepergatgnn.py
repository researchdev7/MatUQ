import os
import sys
import csv
import json
import numpy as np
import ase
from ase import io
from scipy.stats import rankdata
import torch
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import DataLoader, Data, InMemoryDataset
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
from torch_geometric.utils import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    DiffGroupNorm
)

def get_deepergatgnn_train_val_test_loader(dataset, train_indexs=None,
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
        num_workers=num_workers,
        pin_memory=pin_memory,
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

################################################################################
# Pytorch datasets
################################################################################

##Fetch dataset; processes the raw data if specified
def DeeperGATGNNData(data_path, task, config=None):

    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    if os.path.exists(os.path.join(data_path, task, "deepergatgnn_data.pt")) == True:
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
        file_names = ["deepergatgnn_data.pt"]
        return file_names


################################################################################
#  Processing
################################################################################
def create_global_feat(atoms_index_arr):
    comp = np.zeros(108)
    temp = np.unique(atoms_index_arr, return_counts=True)
    for i in range(len(temp[0])):
        comp[temp[0][i]] = temp[1][i] / temp[1].sum()
    return comp.reshape(1, -1)


def process_data(data_path, task, processing_args):
    ##Begin processing data
    print("Processing data to: " + os.path.join(data_path, task))
    assert os.path.exists(data_path), "Data path not found in " + data_path

    ##Load dictionary
    dictionary_file_path = os.path.join(
        data_path, "dictionary_default.json"
    )
    if os.path.exists(dictionary_file_path) == False:
        print("Atom dictionary not found, exiting program...")
        sys.exit()
    else:
        print("Loading atom dictionary from file.")
        atom_dictionary = get_dictionary(dictionary_file_path)

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
        data = Data()

        ##Read in structure file using ase
        ase_crystal = ase.io.read(
            os.path.join(
                data_path, task, structure_id + ".cif"
            )
        )
        data.ase = ase_crystal
        ##Compile structure sizes (# of atoms) and elemental compositions
        if index == 0:
            length = [len(ase_crystal)]
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))

        ##Obtain distance matrix with ase
        distance_matrix = ase_crystal.get_all_distances(mic=True)

        ##Create sparse graph from distance matrix
        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]

        self_loops = True
        if self_loops == True:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
            )
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            distance_matrix_mask = (
                    distance_matrix_trimmed.fill_diagonal_(1) != 0
            ).int()
        elif self_loops == False:
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            distance_matrix_mask = (distance_matrix_trimmed != 0).int()

        data.edge_descriptor = {}
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_matrix_mask

        target = target_data[index][1]
        y = torch.Tensor([float(target)])
        data.y = y

        _atoms_index = ase_crystal.get_atomic_numbers()
        gatgnn_glob_feat = create_global_feat(_atoms_index)
        gatgnn_glob_feat = np.repeat(gatgnn_glob_feat, len(_atoms_index), axis=0)
        data.glob_feat = torch.Tensor(gatgnn_glob_feat).float()

        # pos = torch.Tensor(ase_crystal.get_positions())
        # data.pos = pos
        z = torch.LongTensor(ase_crystal.get_atomic_numbers())
        data.z = z

        ###placeholder for state feature
        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        data.u = u

        data.structure_id = [structure_id]

        if processing_args["verbose"] == "True" and (
                (index + 1) % 500 == 0 or (index + 1) == len(target_data)
        ):
            print("Data processed: ", index + 1, "out of", len(target_data))
            # if index == 0:
            # print(data)
            # print(data.edge_weight, data.edge_attr[0])

        data_list.append(data)

    ##
    n_atoms_max = max(length)
    species = list(set(sum(elements, [])))
    species.sort()

    num_species = len(species)
    if processing_args["verbose"] == "True":
        print(
            "Max structure size: ",
            n_atoms_max,
            "Max number of elements: ",
            num_species,
        )
        print("Unique species:", species)
    crystal_length = len(ase_crystal)
    data.length = torch.LongTensor([crystal_length])

    ##Generate node features
    ##Atom features(node features) from atom dictionary file
    for index in range(0, len(data_list)):
        atom_fea = np.vstack(
            [
                atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                for i in range(len(data_list[index].ase))
            ]
        ).astype(float)
        # print([
        #         atom_dictionary[data_list[index].ase.get_atomic_numbers()[i]]
        #         for i in range(len(data_list[index].ase))
        #     ]);
        # exit()
        data_list[index].x = torch.Tensor(atom_fea)

    ##Adds node degree to node features (appears to improve performance)
    for index in range(0, len(data_list)):
        data_list[index] = OneHotDegree(
            data_list[index], processing_args["graph_max_neighbors"] + 1
        )

    ##Generate edge features
    if processing_args["edge_features"] == "True":

        ##Distance descriptor using a Gaussian basis
        distance_gaussian = GaussianSmearing(
            0, processing_args["graph_max_radius"], processing_args["graph_edge_length"], 0.2
        )
        # print(GetRanges(data_list, 'distance'))
        NormalizeEdge(data_list, "distance")
        # print(GetRanges(data_list, 'distance'))
        for index in range(0, len(data_list)):
            data_list[index].edge_attr = distance_gaussian(
                data_list[index].edge_descriptor["distance"]
            )
            if processing_args["verbose"] == "True" and (
                    (index + 1) % 500 == 0 or (index + 1) == len(target_data)
            ):
                print("Edge processed: ", index + 1, "out of", len(target_data))

    Cleanup(data_list, ["ase", "edge_descriptor"])

    if os.path.isdir(os.path.join(data_path, task)) == False:
        os.mkdir(os.path.join(data_path, task))

    ##Save processed dataset to file
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(data_path, task, "deepergatgnn_data.pt"))

################################################################################
#  Processing sub-functions
################################################################################

##Selects edges with distance threshold and limited number of neighbors
def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr


##Slightly edited version from pytorch geometric to create edge from gaussian basis
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


##Obtain node degree in one-hot representation
def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data


##Obtain dictionary file for elemental features
def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


##Deletes unnecessary data due to slow dataloader
def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


##Get min/max ranges for normalized edges
def GetRanges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max


##Normalizes edges
def NormalizeEdge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
                                                         data.edge_descriptor[descriptor_label] - feature_min
                                                 ) / (feature_max - feature_min)


class GATGNN_GIM1_globalATTENTION(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(GATGNN_GIM1_globalATTENTION, self).__init__()

        self.act = act
        self.fc_layers = fc_layers
        if batch_track_stats == "False":
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True

        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.global_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()

        assert fc_layers > 1, "Need at least 2 fc layer"

        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim + 108, dim)
                self.global_mlp.append(lin)
            else:
                if i != self.fc_layers:
                    lin = torch.nn.Linear(dim, dim)
                else:
                    lin = torch.nn.Linear(dim, 1)
                self.global_mlp.append(lin)

            if self.batch_norm == "True":
                # bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

    def forward(self, x, batch, glbl_x):
        out = torch.cat([x, glbl_x], dim=-1)
        for i in range(0, len(self.global_mlp)):
            if i != len(self.global_mlp) - 1:
                out = self.global_mlp[i](out)
                out = getattr(F, self.act)(out)
            else:
                out = self.global_mlp[i](out)
                out = tg_softmax(out, batch)
        return out

        x = getattr(F, self.act)(self.node_layer1(chunk))
        x = self.atten_layer(x)
        out = tg_softmax(x, batch)
        return out


class GATGNN_AGAT_LAYER(MessagePassing):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2, **kwargs):
        super(GATGNN_AGAT_LAYER, self).__init__(aggr='add', flow='target_to_source', **kwargs)

        self.act = act
        self.fc_layers = fc_layers
        if batch_track_stats == "False":
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True

        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        # FIXED-lines ------------------------------------------------------------
        self.heads = 4
        self.add_bias = True
        self.neg_slope = 0.2

        self.bn1 = nn.BatchNorm1d(self.heads)
        self.W = Parameter(torch.Tensor(dim * 2, self.heads * dim))
        self.att = Parameter(torch.Tensor(1, self.heads, 2 * dim))
        self.dim = dim

        if self.add_bias:
            self.bias = Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # FIXED-lines -------------------------------------------------------------

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        out_i = torch.cat([x_i, edge_attr], dim=-1)
        out_j = torch.cat([x_j, edge_attr], dim=-1)

        out_i = getattr(F, self.act)(torch.matmul(out_i, self.W))
        out_j = getattr(F, self.act)(torch.matmul(out_j, self.W))
        out_i = out_i.view(-1, self.heads, self.dim)
        out_j = out_j.view(-1, self.heads, self.dim)

        alpha = getattr(F, self.act)((torch.cat([out_i, out_j], dim=-1) * self.att).sum(dim=-1))
        alpha = getattr(F, self.act)(self.bn1(alpha))
        alpha = tg_softmax(alpha, edge_index_i)

        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        out_j = (out_j * alpha.view(-1, self.heads, 1)).transpose(0, 1)
        return out_j

    def update(self, aggr_out):
        out = aggr_out.mean(dim=0)
        if self.bias is not None:  out = out + self.bias
        return out


# CGCNN
class DEEP_GATGNN(torch.nn.Module):
    def __init__(
            self,
            data,
            dim1=64,
            dim2=64,
            pre_fc_count=1,
            gc_count=5,
            post_fc_count=1,
            pool="global_add_pool",
            pool_order="early",
            batch_norm="True",
            batch_track_stats="True",
            act="softplus",
            dropout_rate=0.0,
            evidential="False",
            classification=False
    ):
        super(DEEP_GATGNN, self).__init__()

        if batch_track_stats == "False":
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.classification = classification
        self.evidential = evidential
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.out_act = nn.Softplus()
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate

        ##================================
        ## global attention initialization
        self.heads = 4
        self.global_att_LAYER = GATGNN_GIM1_globalATTENTION(dim1, act, batch_norm, batch_track_stats, dropout_rate)
        ##================================

        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 gat layer"
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        if pre_fc_count > 0:
            self.pre_lin_list_E = torch.nn.ModuleList()
            self.pre_lin_list_N = torch.nn.ModuleList()

            for i in range(pre_fc_count):
                if i == 0:
                    lin_N = torch.nn.Linear(data.num_features, dim1)
                    self.pre_lin_list_N.append(lin_N)
                    lin_E = torch.nn.Linear(data.num_edge_features, dim1)
                    self.pre_lin_list_E.append(lin_E)
                else:
                    lin_N = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list_N.append(lin_N)
                    lin_E = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list_E.append(lin_E)

        elif pre_fc_count == 0:
            self.pre_lin_list_N = torch.nn.ModuleList()
            self.pre_lin_list_E = torch.nn.ModuleList()

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = GATGNN_AGAT_LAYER(dim1, act, batch_norm, batch_track_stats, dropout_rate)
            self.conv_list.append(conv)
            ##Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm == "True":
                # bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                bn = DiffGroupNorm(gc_dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            if self.classification:
                self.lin_out = torch.nn.Linear(dim2, 2)
            elif self.evidential == "True":
                self.lin_out = torch.nn.Linear(dim2, 4)
            else:
                self.lin_out = torch.nn.Linear(dim2, 1)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                if self.classification:
                    self.lin_out = torch.nn.Linear(post_fc_dim * 2, 2)
                elif self.evidential == "True":
                    self.lin_out = torch.nn.Linear(post_fc_dim * 2, 4)
                else:
                    self.lin_out = torch.nn.Linear(post_fc_dim * 2, 1)
            else:
                if self.classification:
                    self.lin_out = torch.nn.Linear(post_fc_dim, 2)
                elif self.evidential == "True":
                    self.lin_out = torch.nn.Linear(post_fc_dim, 4)
                else:
                    self.lin_out = torch.nn.Linear(post_fc_dim, 1)


        ##Set up set2set pooling (if used)
        ##Should processing_setps be a hypereparameter?
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            if self.classification:
                self.set2set = Set2Set(2, processing_steps=3, num_layers=1)
                self.lin_out_2 = torch.nn.Linear(2 * 2, 2)
            elif self.evidential == "True":
                self.set2set = Set2Set(4, processing_steps=3, num_layers=1)
                self.lin_out_2 = torch.nn.Linear(4 * 2, 4)
            else:
                self.set2set = Set2Set(1, processing_steps=3, num_layers=1)
                self.lin_out_2 = torch.nn.Linear(1 * 2, 1)

        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.cdropout = nn.Dropout()

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, data):

        min_val = 1e-6
        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list_N)):
            if i == 0:
                out_x = self.pre_lin_list_N[i](data.x)
                out_x = getattr(F, 'leaky_relu')(out_x, 0.2)
                out_e = self.pre_lin_list_E[i](data.edge_attr)
                out_e = getattr(F, 'leaky_relu')(out_e, 0.2)
            else:
                out_x = self.pre_lin_list_N[i](out_x)
                out_x = getattr(F, self.act)(out_x)
                out_e = self.pre_lin_list_E[i](out_e)
                out_e = getattr(F, 'leaky_relu')(out_e, 0.2)
        prev_out_x = out_x

        ##GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list_N) == 0 and i == 0:
                if self.batch_norm == "True":
                    out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    out_x = self.bn_list[i](out_x)
                else:
                    out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
            else:
                if self.batch_norm == "True":
                    out_x = self.conv_list[i](out_x, data.edge_index, out_e)
                    out_x = self.bn_list[i](out_x)
                else:
                    out_x = self.conv_list[i](out_x, data.edge_index, out_e)
            out_x = torch.add(out_x, prev_out_x)
            out_x = F.dropout(out_x, p=self.dropout_rate, training=self.training)
            prev_out_x = out_x

        # exit()

        ##GLOBAL attention
        # print(out_x.shape)
        # exit()
        out_a = self.global_att_LAYER(out_x, data.batch, data.glob_feat)
        out_x = out_x * out_a

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                out_x = self.set2set(out_x, data.batch)
            else:
                out_x = getattr(torch_geometric.nn, self.pool)(out_x, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out_x = self.post_lin_list[i](out_x)
                out_x = getattr(F, self.act)(out_x)
            out_x = self.dropout(out_x)
            out = self.lin_out(out_x)

        elif self.pool_order == "late":
            for i in range(0, len(self.post_lin_list)):
                out_x = self.post_lin_list[i](out_x)
                out_x = getattr(F, self.act)(out_x)
            out_x = self.dropout(out_x)
            out = self.lin_out(out_x)
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        if self.classification:
            out = self.logsoftmax(out)
        if self.evidential=="True":
            if out.shape[0] == 4:
                out = torch.unsqueeze(out, 0)
            out = out.view(out.shape[0], -1, 4)
            mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(out, 1, dim=-1)]
            return mu, self.out_act(logv)+ min_val, self.out_act(logalpha)+ min_val + 1, self.out_act(logbeta)+ min_val
        else:
            return torch.squeeze(out)
