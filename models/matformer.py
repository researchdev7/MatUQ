"""Implementation based on the template of ALIGNN."""
import copy
import os
import csv
import pickle
import sys

import math
from math import pi as PI
import json
import itertools
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Sequence
from pathlib import Path
from scipy.optimize import brentq
from scipy import special as sp
import sympy as sym
import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.core.specie import chem_data, get_node_attributes
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.data.batch import Batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import gather_csr, scatter, segment_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes


def canonize_edge(
        src_id,
        dst_id,
        src_image,
        dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges_submit(
        atoms=None,
        cutoff=8,
        max_neighbors=12,
        id=None,
        use_canonize=False,
        use_lattice=False,
        use_angle=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )

    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 1, 0])))

    return edges


def pair_nearest_neighbor_edges(
        atoms=None,
        pair_wise_distances=6,
        use_lattice=False,
        use_angle=False,
):
    """Construct pairwise k-fully connected edge list."""
    smallest = pair_wise_distances
    lattice_list = torch.as_tensor(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]).float()

    lattice = torch.as_tensor(atoms.lattice_mat).float()
    pos = torch.as_tensor(atoms.cart_coords)
    atom_num = pos.size(0)
    lat = atoms.lattice
    radius_needed = min(lat.a, lat.b, lat.c) * (smallest / 2 - 1e-9)
    r_a = (np.floor(radius_needed / lat.a) + 1).astype(np.int)
    r_b = (np.floor(radius_needed / lat.b) + 1).astype(np.int)
    r_c = (np.floor(radius_needed / lat.c) + 1).astype(np.int)
    period_list = np.array([l for l in itertools.product(
        *[list(range(-r_a, r_a + 1)), list(range(-r_b, r_b + 1)), list(range(-r_c, r_c + 1))])])
    period_list = torch.as_tensor(period_list).float()
    n_cells = period_list.size(0)
    offset = torch.matmul(period_list, lattice).view(n_cells, 1, 3)
    expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offset).transpose(0, 1).contiguous()
    dist = (pos.unsqueeze(1).unsqueeze(1) - expand_pos.unsqueeze(
        0))  # [n, 1, 1, 3] - [1, n, n_cell, 3] -> [n, n, n_cell, 3]
    dist2, index = torch.sort(dist.norm(dim=-1), dim=-1, stable=True)
    max_value = dist2[:, :, smallest - 1]  # [n, n]
    mask = (dist.norm(dim=-1) <= max_value.unsqueeze(-1))  # [n, n, n_cell]
    shift = torch.matmul(lattice_list, lattice).repeat(atom_num, 1)
    shift_src = torch.arange(atom_num).unsqueeze(-1).repeat(1, lattice_list.size(0))
    shift_src = torch.cat([shift_src[i, :] for i in range(shift_src.size(0))])

    indices = torch.where(mask)
    dist_target = dist[indices]
    u, v, _ = indices
    if use_lattice:
        u = torch.cat((u, shift_src), dim=0)
        v = torch.cat((v, shift_src), dim=0)
        dist_target = torch.cat((dist_target, shift), dim=0)
        assert u.size(0) == dist_target.size(0)

    return u, v, dist_target


def build_undirected_edgedata(
        atoms=None,
        edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            # if np.linalg.norm(d)!=0:
            # print ('jv',dst_image,d)
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())

    return u, v, r

def pyg_compute_bond_cosines(lg):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

def pyg_compute_bond_angle(lg):
    """Compute bond angle from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    a = (r1 * r2).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(r1, r2).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return angle

class PygGraph(object):
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels


    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = True,
        use_lattice: bool = False,
        use_angle: bool = False,
        lineControl=True,
    ):
        # print('id',id)
        if neighbor_strategy == "k-nearest":
            edges = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            u, v, r = build_undirected_edgedata(atoms, edges)
        elif neighbor_strategy == "pairwise-k-nearest":
            u, v, r = pair_nearest_neighbor_edges(
                atoms=atoms,
                pair_wise_distances=2,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
        g = Data(x=node_features, edge_index=edge_index, edge_attr=r)

        if compute_line_graph:
            if lineControl == True:
                linegraph_trans = LineGraph(force_directed=True)
                g_new = Data()
                g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
                try:
                    lg = linegraph_trans(g)
                except Exception as exp:
                    print(g.x, g.edge_attr, exp)
                    pass
                lg.edge_attr = pyg_compute_bond_cosines(lg) # old cosine emb
                # lg.edge_attr = pyg_compute_bond_angle(lg)
                return [g_new, lg]

            else:
                g.edge_attr = g.edge_attr.float()
                lg = g
                return [g, lg]
        else:
            return g


class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        graphs: Sequence[Data],
        line_graphs: Sequence[Data],
        targets=None,
        transform=None,
        line_graph=False,
        classification=False,
        ids=None,
    ):
        """Pytorch Dataset for atomistic graphs.

        """
        self.graphs = graphs
        self.line_graphs = line_graphs
        self.line_graph = line_graph
        self.ids = ids
        self.labels = targets
        self.transform = transform

        if classification:
            self.labels = torch.stack(self.labels, dim=0).view(-1).long()
            print("Classification dataset.", self.labels)

    def __len__(self):
        """Get length."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]
        cif_id = self.ids[idx]
        if self.transform:
            g = self.transform(g)

        if self.line_graph:
            return g, self.line_graphs[idx], label, cif_id

        return g, label, cif_id

    @staticmethod
    def collate(samples):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels, batch_cif_ids = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.stack(labels, dim=0), batch_cif_ids

    @staticmethod
    def collate_line_graph(samples):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels, batch_cif_ids = map(list, zip(*samples))
        lattice = copy.deepcopy(labels)
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        return (batched_graph, batched_line_graph, torch.cat([i for i in lattice])), torch.stack(labels, dim=0), batch_cif_ids

def load_matformer_data(root_dir: str='data/', task: str=None, config: dict=None):
    assert os.path.exists(root_dir), 'root_dir does not exist!'
    id_prop_file = os.path.join(root_dir, task, 'targets.csv')
    assert os.path.exists(id_prop_file), 'targets.csv does not exist!'
    if os.path.exists(os.path.join(root_dir, task, "matformer_data.pkl")) == True:
        with open(os.path.join(root_dir, task, "matformer_data.pkl"), 'rb') as f:
            data = pickle.load(f)
    else:
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            id_prop_data = [row for row in reader]
        data = []
        for d in id_prop_data:
            cif_id, target = d
            crystal = Atoms.from_cif(os.path.join(root_dir, task, cif_id + '.cif'))
            crystal = crystal.to_dict()
            structure = (
                Atoms.from_dict(crystal) if isinstance(crystal, dict) else crystal
            )
            g = PygGraph.atom_dgl_multigraph(
                structure,
                neighbor_strategy="k-nearest",
                cutoff=config["cutoff"],
                atom_features="cgcnn",
                max_neighbors=config["max_neighbors"],
                compute_line_graph=True,
                use_canonize=True,
                use_lattice= True,
                use_angle= False,
                id=cif_id,
            )
            data.append([g, torch.Tensor([float(target)]), cif_id])
        with open(os.path.join(root_dir, task, "matformer_data.pkl"), 'wb') as f:
            pickle.dump(data, f)

    return data

def get_matformer_train_val_test_loader(dataset, train_indexs=None,
                              val_indexs=None, test_indexs=None,
                              batch_size=256, return_test=True,
                              num_workers=0, pin_memory=False):

    total_size = len(dataset)
    train_indice = []
    val_indice = []
    test_indice = []
    for i in range(total_size):
        cif_id = dataset[i][2]
        _cif_id = int(cif_id)
        if _cif_id in train_indexs:
            train_indice.append(i)
        elif _cif_id in val_indexs:
            val_indice.append(i)
        elif _cif_id in test_indexs:
            test_indice.append(i)
        else:
            print("Can't find data which cif_id is %d in dataset", _cif_id)


    train_dataset = [dataset[x][0][0] for x in train_indice]
    train_line_graphs = [dataset[x][0][1] for x in train_indice]
    train_targets = [dataset[x][1] for x in train_indice]
    train_ids = [dataset[x][2] for x in train_indice]

    val_dataset = [dataset[x][0][0] for x in val_indice]
    val_line_graphs = [dataset[x][0][1] for x in val_indice]
    val_targets = [dataset[x][1] for x in val_indice]
    val_ids = [dataset[x][2] for x in val_indice]

    test_dataset = [dataset[x][0][0] for x in test_indice]
    test_line_graphs = [dataset[x][0][1] for x in test_indice]
    test_targets = [dataset[x][1] for x in test_indice]
    test_ids = [dataset[x][2] for x in test_indice]

    train_data = PygStructureDataset(
        train_dataset,
        train_line_graphs,
        targets=train_targets,
        line_graph=True,
        ids=train_ids,
    )
    val_data = PygStructureDataset(
        val_dataset,
        val_line_graphs,
        targets=val_targets,
        line_graph=True,
        ids=val_ids,
    )
    test_data = PygStructureDataset(
        test_dataset,
        test_line_graphs,
        targets=test_targets,
        line_graph=True,
        ids=test_ids,
    )

    collate_fn = train_data.collate_line_graph

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if return_test:
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)
def Jn_zeros(n, k):
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i, ))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj

def spherical_bessel_formulas(n):
    x = sym.symbols('x')

    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x)**i)]
        a = sym.simplify(b)
    return f

def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1)**2]
        normalizer_tmp = 1 / np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] *
                             f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis

def sph_harm_prefactor(k, m):
    return ((2 * k + 1) * np.math.factorial(k - abs(m)) /
            (4 * np.pi * np.math.factorial(k + abs(m))))**0.5


def associated_legendre_polynomials(k, zero_m_only=True):
    z = sym.symbols('z')
    P_l_m = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            P_l_m[j][0] = sym.simplify(((2 * j - 1) * z * P_l_m[j - 1][0] -
                                        (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] -
                         (i + j - 1) * P_l_m[j - 2][i]) / (j - i))

    return P_l_m
def real_sph_harm(l, zero_m_only=False, spherical_coordinates=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to order l (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        x = sym.symbols('x')
        y = sym.symbols('y')
        S_m = [x*0]
        C_m = [1+0*x]
        # S_m = [0]
        # C_m = [1]
        for i in range(1, l):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x*S_m[i-1] + y*C_m[i-1]]
            C_m += [x*C_m[i-1] - y*S_m[i-1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))

    Y_func_l_m = [['0']*(2*j + 1) for j in range(l)]
    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


class angle_emb_mp(torch.nn.Module):
    def __init__(self, num_spherical=3, num_radial=30, cutoff=8.0,
                 envelope_exponent=5):
        super(angle_emb_mp, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

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
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )


class MatformerConv(MessagePassing):
    _alpha: OptTensor
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(MatformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], out_channels,
                                      bias=bias)
            self.lin_concate = nn.Linear(heads * out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(nn.Linear(out_channels * 3, out_channels), nn.LayerNorm(out_channels))
        # self.msg_layer = nn.Linear(out_channels * 3, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        # self.bn = nn.BatchNorm1d(out_channels * heads)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.concat:
            self.lin_concate.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.concat:
            out = self.lin_concate(out)

        out = F.silu(self.bn(out))  # after norm and silu

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
        query_i = torch.cat((query_i, query_i, query_i), dim=-1)
        key_j = torch.cat((key_i, key_j, edge_attr), dim=-1)
        alpha = (query_i * key_j) / math.sqrt(self.out_channels * 3)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.cat((value_i, value_j, edge_attr), dim=-1)
        out = self.lin_msg_update(out) * self.sigmoid(
            self.layer_norm(alpha.view(-1, self.heads, 3 * self.out_channels)))
        out = self.msg_layer(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class Matformer(nn.Module):
    """att pyg implementation."""

    def __init__(
        self,
        conv_layers: int = 5,
        edge_layers: int = 0,
        atom_input_features: int = 92,
        edge_features: int = 128,
        triplet_input_features: int = 40,
        node_features: int = 128,
        fc_features: int = 128,
        output_features: int = 1,
        node_layer_head: int = 4,
        edge_layer_head: int = 4,
        use_angle: bool = False,
        angle_lattice: bool = False,
        evidential: str = "False",
        classification: bool = False,
    ):
        """Set up att modules."""
        super(Matformer, self).__init__()
        self.classification = classification
        self.evidential = evidential
        self.use_angle = use_angle
        self.atom_embedding = nn.Linear(
            atom_input_features, node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=edge_features,
            ),
            nn.Linear(edge_features, node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features),
        )
        self.angle_lattice = angle_lattice
        if self.angle_lattice:  ## module not used
            print('use angle lattice')
            self.lattice_rbf = nn.Sequential(
                RBFExpansion(
                    vmin=0,
                    vmax=8.0,
                    bins=edge_features,
                ),
                nn.Linear(edge_features, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )

            self.lattice_angle = nn.Sequential(
                RBFExpansion(
                    vmin=-1,
                    vmax=1.0,
                    bins=triplet_input_features,
                ),
                nn.Linear(triplet_input_features, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )

            self.lattice_emb = nn.Sequential(
                nn.Linear(node_features * 6, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )

            self.lattice_atom_emb = nn.Sequential(
                nn.Linear(node_features * 2, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )

        self.edge_init = nn.Sequential(  ## module not used
            nn.Linear(3 * node_features, node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features)
        )

        self.sbf = angle_emb_mp(num_spherical=3, num_radial=40, cutoff=8.0)  ## module not used

        self.angle_init_layers = nn.Sequential(  ## module not used
            nn.Linear(120, node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features)
        )

        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=node_features, out_channels=node_features,
                              heads=node_layer_head, edge_dim=node_features)
                for _ in range(conv_layers)
            ]
        )

        self.edge_update_layers = nn.ModuleList(  ## module not used
            [
                MatformerConv(in_channels=node_features, out_channels=node_features,
                              heads=edge_layer_head, edge_dim=node_features)
                for _ in range(edge_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(node_features, fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        elif self.evidential == "True":
            self.fc_out = nn.Linear(fc_features, 4)
            self.out_act = nn.Softplus()
        else:
            self.fc_out = nn.Linear(
                fc_features, output_features
            )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, data) -> torch.Tensor:
        min_val = 1e-6
        data, ldata, lattice = data
        # initial node features: atom feature network...

        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)

        edge_features = self.rbf(edge_feat)
        if self.angle_lattice:  ## module not used
            lattice_len = torch.norm(lattice, dim=-1)  # batch * 3 * 1
            lattice_edge = self.lattice_rbf(lattice_len.view(-1)).view(-1, 3 * 128)  # batch * 3 * 128
            cos1 = self.lattice_angle(torch.clamp(torch.sum(lattice[:, 0, :] * lattice[:, 1, :], dim=-1) / (
                        torch.norm(lattice[:, 0, :], dim=-1) * torch.norm(lattice[:, 1, :], dim=-1)), -1, 1).unsqueeze(
                -1)).view(-1, 128)
            cos2 = self.lattice_angle(torch.clamp(torch.sum(lattice[:, 0, :] * lattice[:, 2, :], dim=-1) / (
                        torch.norm(lattice[:, 0, :], dim=-1) * torch.norm(lattice[:, 2, :], dim=-1)), -1, 1).unsqueeze(
                -1)).view(-1, 128)
            cos3 = self.lattice_angle(torch.clamp(torch.sum(lattice[:, 1, :] * lattice[:, 2, :], dim=-1) / (
                        torch.norm(lattice[:, 1, :], dim=-1) * torch.norm(lattice[:, 2, :], dim=-1)), -1, 1).unsqueeze(
                -1)).view(-1, 128)
            lattice_emb = self.lattice_emb(torch.cat((lattice_edge, cos1, cos2, cos3), dim=-1))
            node_features = self.lattice_atom_emb(torch.cat((node_features, lattice_emb[data.batch]), dim=-1))

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[4](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        if self.angle_lattice:
            # features *= F.sigmoid(lattice_emb)
            features += lattice_emb

        # features = F.softplus(features)
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


