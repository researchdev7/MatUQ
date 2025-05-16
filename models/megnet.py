"""Tools to construct a dataset of DGL graphs."""

from __future__ import annotations

import copy
import logging
import json
import os
import csv
import random
from functools import partial
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from pymatgen.core.structure import Structure
import dgl
from dgl.data import DGLDataset
from dgl.nn import AvgPooling
from dgl.data.utils import load_graphs, save_graphs, Subset
from dgl.dataloading import GraphDataLoader
from tqdm import trange

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph
from matgl.layers import MLP, ActivationFunction, BondExpansion, EmbeddingBlock, MEGNetBlock
from matgl.utils.io import IOMixIn

if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__file__)


def megnet_collate_fn_graph(batch, include_line_graph: bool = False):
    """Merge a list of dgl graphs to form a batch."""
    line_graphs = None
    if include_line_graph:
        graphs, lattices, line_graphs, state_attr, targets, ids = map(list, zip(*batch))
    else:
        graphs, lattices, state_attr, targets, ids = map(list, zip(*batch))
    g = dgl.batch(graphs)
    targets = torch.stack(targets, dim=0)

    state_attr = torch.stack(state_attr)
    lat = lattices[0] if g.batch_size == 1 else torch.squeeze(torch.stack(lattices))
    if include_line_graph:
        l_g = dgl.batch(line_graphs)
        return g, lat, l_g, state_attr, targets
    return (g, lat, state_attr), targets, ids

def load_megnet_data(root_dir: str='data/', task: str=None, seed: int=123):
    assert os.path.exists(root_dir), 'root_dir does not exist!'
    id_prop_file = os.path.join(root_dir, task, 'targets.csv')
    assert os.path.exists(id_prop_file), 'targets.csv does not exist!'
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]
    random.seed(seed)
    random.shuffle(id_prop_data)
    cif_ids = []
    targets = []
    crystals = []
    for d in id_prop_data:
        cif_id, target = d
        cif_ids.append(cif_id)
        targets.append(torch.Tensor([float(target)]))
        crystal = Structure.from_file(os.path.join(root_dir, task,
                                                   cif_id + '.cif'))
        crystals.append(crystal)
    return cif_ids, targets, crystals


class MEGNetDataset(DGLDataset):
    """Create a dataset including dgl graphs."""

    def __init__(
        self,
        filename: str = "dgl_graph.bin",
        filename_lattice: str = "lattice.pt",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_state_attr: str = "state_attr.pt",
        filename_labels: str = "labels.pt",
        filename_idx: str = "idx.pt",
        include_line_graph: bool = False,
        converter: GraphConverter | None = None,
        threebody_cutoff: float | None = None,
        directed_line_graph: bool = False,
        structures: list | None = None,
        targets: list | None = None,
        ids: list[str] | None = None,
        name: str = "MEGNetDataset",
        task: str | None = None,
        graph_labels: list[int | float] | None = None,
        clear_processed: bool = False,
        save_cache: bool = False,
        save_dir: str | None = None,
    ):
        """
        Args:
            filename: file name for storing dgl graphs.
            filename_lattice: file name for storing lattice matrixs.
            filename_line_graph: file name for storing dgl line graphs.
            filename_state_attr: file name for storing state attributes.
            filename_labels: file name for storing labels.
            include_line_graph: whether to include line graphs.
            converter: dgl graph converter.
            threebody_cutoff: cutoff for three body.
            directed_line_graph (bool): Whether to create a directed line graph (CHGNet), or an
                undirected 3body line graph (M3GNet)
                Default: False (for M3GNet)
            structures: Pymatgen structure.
            targets: targets.
            ids: id set of dataset
            name: name of dataset.
            graph_labels: state attributes.
            clear_processed: Whether to clear the stored structures after processing into graphs. Structures
                are not really needed after the conversion to DGL graphs and can take a significant amount of memory.
                Setting this to True will delete the structures from memory.
            save_cache: whether to save the processed dataset. The dataset can be reloaded from save_dir
                Default: True
            raw_dir : str specifying the directory that will store the downloaded data or the directory that already
                stores the input data.
                Default: ~/.dgl/
            save_dir : directory to save the processed dataset. Default: same as raw_dir.
        """
        self.filename = filename
        self.filename_lattice = filename_lattice
        self.filename_line_graph = filename_line_graph
        self.filename_state_attr = filename_state_attr
        self.filename_labels = filename_labels
        self.filename_idx = filename_idx
        self.include_line_graph = include_line_graph
        self.converter = converter
        self.structures = structures or []
        self.targets = targets or []
        self.ids = ids or []
        self.threebody_cutoff = threebody_cutoff
        self.directed_line_graph = directed_line_graph
        self.graph_labels = graph_labels
        self.clear_processed = clear_processed
        self.save_cache = save_cache
        save_task_dir = save_dir + '/' + task + '/'
        super().__init__(name=name, raw_dir=save_task_dir, save_dir=save_task_dir)

    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not."""
        files_to_check = [
            self.filename,
            self.filename_lattice,
            self.filename_state_attr,
            self.filename_labels,
            self.filename_idx,
        ]
        if self.include_line_graph:
            files_to_check.append(self.filename_line_graph)
        return all(os.path.exists(os.path.join(self.save_path, f)) for f in files_to_check)

    def process(self):
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = len(self.structures)  # type: ignore
        graphs, lattices, line_graphs, state_attrs = [], [], [], []

        for idx in trange(num_graphs):
            structure = self.structures[idx]  # type: ignore
            graph, lattice, state_attr = self.converter.get_graph(structure)  # type: ignore
            graphs.append(graph)
            lattices.append(lattice)
            state_attrs.append(state_attr)
            graph.ndata["pos"] = torch.tensor(structure.cart_coords)
            graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["bond_vec"] = bond_vec
            graph.edata["bond_dist"] = bond_dist
            if self.include_line_graph:
                line_graph = create_line_graph(graph, self.threebody_cutoff, directed=self.directed_line_graph)  # type: ignore
                for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                    line_graph.ndata.pop(name)
                line_graphs.append(line_graph)
            graph.ndata.pop("pos")
            graph.edata.pop("pbc_offshift")
        if self.graph_labels is not None:
            state_attrs = torch.tensor(self.graph_labels).long()
        else:
            state_attrs = torch.tensor(np.array(state_attrs), dtype=matgl.float_th)

        if self.clear_processed:
            del self.structures
            self.structures = []

        self.graphs = graphs
        self.lattices = lattices
        self.state_attr = state_attrs
        if self.include_line_graph:
            self.line_graphs = line_graphs
            return self.graphs, self.lattices, self.line_graphs, self.state_attr
        return self.graphs, self.lattices, self.state_attr

    def save(self):
        """Save dgl graphs and labels to self.save_path."""
        if self.save_cache is False:
            return

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.targets:
            torch.save(self.targets, os.path.join(self.save_path, self.filename_labels))
        torch.save(self.ids, os.path.join(self.save_path, self.filename_idx))
        save_graphs(os.path.join(self.save_path, self.filename), self.graphs)
        torch.save(self.lattices, os.path.join(self.save_path, self.filename_lattice))
        torch.save(self.state_attr, os.path.join(self.save_path, self.filename_state_attr))
        if self.include_line_graph:
            save_graphs(os.path.join(self.save_path, self.filename_line_graph), self.line_graphs)

    def load(self):
        """Load dgl graphs from files."""
        self.graphs, _ = load_graphs(os.path.join(self.save_path, self.filename))
        self.lattices = torch.load(os.path.join(self.save_path, self.filename_lattice))
        if self.include_line_graph:
            self.line_graphs, _ = load_graphs(os.path.join(self.save_path, self.filename_line_graph))
        self.state_attr = torch.load(os.path.join(self.save_path, self.filename_state_attr))
        self.targets = torch.load(os.path.join(self.save_path, self.filename_labels))
        self.ids = torch.load(os.path.join(self.save_path, self.filename_idx))

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        items = [
            self.graphs[idx],
            self.lattices[idx],
            self.state_attr[idx],
            self.targets[idx],
            self.ids[idx],
        ]
        if self.include_line_graph:
            items.insert(2, self.line_graphs[idx])
        return tuple(items)

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)

def get_megnet_train_val_test_loader(dataset, train_indexs=None,
                              val_indexs=None, test_indexs=None,
                              collate_fn=default_collate, **kwargs):
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
    num_workers: int

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    """
    total_size = len(dataset)
    train_indice = []
    val_indice = []
    test_indice = []
    for i in range(total_size):
        _, _, _, _, cif_id = dataset[i]
        _cif_id = int(cif_id)
        if _cif_id in train_indexs:
            train_indice.append(i)
        elif _cif_id in val_indexs:
            val_indice.append(i)
        elif _cif_id in test_indexs:
            test_indice.append(i)
        else:
            print("Can't find data which cif_id is %d in dataset", _cif_id)

    train_data = copy.deepcopy(Subset(dataset, train_indice))
    val_data = copy.deepcopy(Subset(dataset, val_indice))
    test_data = copy.deepcopy(Subset(dataset, test_indice))

    train_loader = GraphDataLoader(train_data, shuffle=True, collate_fn=collate_fn, **kwargs)
    val_loader = GraphDataLoader(val_data, shuffle=False, collate_fn=collate_fn, **kwargs)
    test_loader = GraphDataLoader(test_data, shuffle=False, collate_fn=collate_fn, **kwargs)

    return train_loader, val_loader, test_loader

class MEGNet(nn.Module, IOMixIn):
    """DGL implementation of MEGNet."""

    __version__ = 1

    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_state_embedding: int = 2,
        ntypes_state: int | None = None,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        activation_type: str = "softplus2",
        is_classification: bool = False,
        evidential: str = "False",
        include_state: bool = True,
        dropout: float = 0.0,
        mc_dropout: float = 0.1,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        bond_expansion: BondExpansion | None = None,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        **kwargs,
    ):
        """Useful defaults for all arguments have been specified based on MEGNet formation energy model.

        Args:
            dim_node_embedding: Dimension of node embedding.
            dim_edge_embedding: Dimension of edge embedding.
            dim_state_embedding: Dimension of state embedding.
            ntypes_state: Number of state types.
            nblocks: Number of blocks.
            hidden_layer_sizes_input: Architecture of dense layers before the graph convolution
            hidden_layer_sizes_conv: Architecture of dense layers for message and update functions
            nlayers_set2set: Number of layers in Set2Set layer
            niters_set2set: Number of iterations in Set2Set layer
            hidden_layer_sizes_output: Architecture of dense layers for concatenated features after graph convolution
            activation_type: Activation used for non-linearity
            is_classification: Whether this is classification task or not
            layer_node_embedding: Architecture of embedding layer for node attributes
            layer_edge_embedding: Architecture of embedding layer for edge attributes
            layer_state_embedding: Architecture of embedding layer for state attributes
            include_state: Whether the state embedding is included
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according to
                a Bernoulli distribution. Defaults to 0, i.e., no dropout.
            element_types: Elements included in the training set
            bond_expansion: Gaussian expansion for edge attributes
            cutoff: cutoff for forming bonds
            gauss_width: width of Gaussian function for bond expansion
            **kwargs: For future flexibility. Not used at the moment.
        """
        super().__init__()

        self.save_args(locals(), kwargs)

        self.element_types = element_types or DEFAULT_ELEMENTS
        self.cutoff = cutoff
        self.bond_expansion = bond_expansion or BondExpansion(
            rbf_type="Gaussian", initial=0.0, final=cutoff, num_centers=dim_edge_embedding, width=gauss_width
        )

        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.readout = AvgPooling()
        self.out_activation = nn.Softplus()
        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=len(self.element_types),
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)

        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]
        block_args = {
            "conv_hiddens": hidden_layer_sizes_conv,
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }
        # first block
        blocks = [MEGNetBlock(dims=[dim_blocks_in], **block_args)] + [  # type: ignore
            MEGNetBlock(dims=[dim_blocks_out, *hidden_layer_sizes_input], **block_args)  # type: ignore
            for _ in range(nblocks - 1)
        ]

        self.blocks = nn.ModuleList(blocks)

        self.evidential = evidential
        if evidential == "True":
            dims = [2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 4]
        else:
            dims = [2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1]

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=True)
                                      for i in range(len(dims)-2)])
        self.acts = nn.ModuleList([activation
                                             for _ in range(len(dims)-2)])

        self.output_proj = nn.Linear(dims[-2],dims[-1],bias=True)

        self.dropout = nn.Dropout(mc_dropout) if mc_dropout else None

        self.is_classification = is_classification
        self.include_state_embedding = include_state

    def forward(self, g: dgl.DGLGraph, state_attr: torch.Tensor | None = None, **kwargs):
        """Forward pass of MEGnet. Executes all blocks.

        Args:
            g (dgl.DGLGraph): DGL graphs
            state_attr (torch.Tensor): State attributes
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            Prediction
        """
        min_val = 1e-6
        node_attr = g.ndata["node_type"]
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec
        g.edata["bond_dist"] = bond_dist
        edge_attr = self.bond_expansion(g.edata["bond_dist"])
        node_feat, edge_feat, state_feat = self.embedding(node_attr, edge_attr, state_attr)
        edge_feat = self.edge_encoder(edge_feat)
        node_feat = self.node_encoder(node_feat)
        state_feat = self.state_encoder(state_feat)

        for block in self.blocks:
            output = block(g, edge_feat, node_feat, state_feat)
            edge_feat, node_feat, state_feat = output

        node_vec = self.readout(g, node_feat)
        g.edata['f'] = edge_feat
        edge_vec = dgl.readout_edges(g, 'f', op='mean')

        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        state_feat = torch.squeeze(state_feat)

        vec = torch.hstack([node_vec, edge_vec, state_feat])

        for fc, act in zip(self.fcs, self.acts):
            vec = act(fc(vec))

        if self.dropout:
            vec = self.dropout(vec)  # pylint: disable=E1102

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        if self.evidential=="True":
            if output.shape[0] == 4:
                output = torch.unsqueeze(output, 0)
            output = output.view(output.shape[0], -1, 4)
            mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(output, 1, dim=-1)]
            return mu, self.out_activation(logv) + min_val, self.out_activation(logalpha) + min_val+ 1, self.out_activation(logbeta)+ min_val
        else:
            return torch.squeeze(output)

    def predict_structure(
        self,
        structure,
        state_attr: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_attr (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)
        g, lat, state_attr_default = graph_converter.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        if state_attr is None:
            state_attr = torch.tensor(state_attr_default)
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["edge_attr"] = self.bond_expansion(bond_dist)
        return self(g=g, state_attr=state_attr).detach()

