"""Implementation based on the template of Matformer."""
import os
import sys
import csv
import pickle
from typing import Tuple, Optional, Union, Sequence
import math
import numpy as np
from collections import defaultdict
from jarvis.core.atoms import Atoms
from jarvis.core.specie import chem_data, get_node_attributes
from e3nn import o3
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor

def same_line(a, b):
    a_new = a / (sum(a ** 2) ** 0.5)
    b_new = b / (sum(b ** 2) ** 0.5)
    flag = False
    if abs(sum(a_new * b_new) - 1.0) < 1e-5:
        flag = True
    elif abs(sum(a_new * b_new) + 1.0) < 1e-5:
        flag = True
    else:
        flag = False
    return flag

def same_plane(a, b, c):
    flag = False
    if abs(np.dot(np.cross(a, b), c)) < 1e-5:
        flag = True
    return flag

def angle_from_array(a, b, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    assert a_new.shape == a.shape
    value = sum(a_new * b_new)
    length = (sum(a_new ** 2) ** 0.5) * (sum(b_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return angle / np.pi * 180.0

def correct_coord_sys(a, b, c, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    c_new = np.dot(c, lattice)
    assert a_new.shape == a.shape
    plane_vec = np.cross(a_new, b_new)
    value = sum(plane_vec * c_new)
    length = (sum(plane_vec ** 2) ** 0.5) * (sum(c_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return (angle / np.pi * 180.0 <= 90.0)

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
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)

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
            use_lattice=use_lattice,
        )

    edges = defaultdict(set)
    # lattice correction process
    r_cut = max(lat.a, lat.b, lat.c) + 1e-2
    all_neighbors = atoms.get_all_neighbors(r=r_cut)
    neighborlist = all_neighbors[0]
    neighborlist = sorted(neighborlist, key=lambda x: x[2])
    ids = np.array([nbr[1] for nbr in neighborlist])
    images = np.array([nbr[3] for nbr in neighborlist])
    images = images[ids == 0]
    lat1 = images[0]
    # finding lat2
    start = 1
    for i in range(start, len(images)):
        lat2 = images[i]
        if not same_line(lat1, lat2):
            start = i
            break
    # finding lat3
    for i in range(start, len(images)):
        lat3 = images[i]
        if not same_plane(lat1, lat2, lat3):
            break
    # find the invariant corner
    if angle_from_array(lat1, lat2, lat.matrix) > 90.0:
        lat2 = - lat2
    if angle_from_array(lat1, lat3, lat.matrix) > 90.0:
        lat3 = - lat3
    # find the invariant coord system
    if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
        lat1 = - lat1
        lat2 = - lat2
        lat3 = - lat3

    # if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
    #     print(lat1, lat2, lat3)
    # lattice correction end
    for site_idx, neighborlist in enumerate(all_neighbors_now):

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
            edges[(site_idx, site_idx)].add(tuple(lat1))
            edges[(site_idx, site_idx)].add(tuple(lat2))
            edges[(site_idx, site_idx)].add(tuple(lat3))

    return edges, lat1, lat2, lat3

def build_undirected_edgedata(
    atoms=None,
    edges={},
    a=None,
    b=None,
    c=None,
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r, l, nei, angle, atom_lat = [], [], [], [], [], [], []
    v1, v2, v3 = atoms.lattice.cart_coords(a), atoms.lattice.cart_coords(b), atoms.lattice.cart_coords(c)
    # atom_lat.append([v1, v2, v3, -v1, -v2, -v3])
    atom_lat.append([v1, v2, v3])
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                # nei.append([v1, v2, v3, -v1, -v2, -v3])
                nei.append([v1, v2, v3])
                # angle.append([compute_bond_cosine(dd, v1), compute_bond_cosine(dd, v2), compute_bond_cosine(dd, v3)])

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())
    l = torch.tensor(l).type(torch.int)
    nei = torch.tensor(np.array(nei)).type(torch.get_default_dtype())
    atom_lat = torch.tensor(np.array(atom_lat)).type(torch.get_default_dtype())
    # nei_angles = torch.tensor(angle).type(torch.get_default_dtype())
    return u, v, r, l, nei, atom_lat

def get_attribute_lookup(atom_features: str = "cgcnn"):
    """Build a lookup array indexed by atomic number."""
    max_z = max(v["Z"] for v in chem_data.values())

    # get feature shape (referencing Carbon)
    template = get_node_attributes("C", atom_features)

    features = np.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features)

        if x is not None:
            features[z, :] = x

    return features

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
        cutoff=4.0,
        max_neighbors=12,
        atom_features="cgcnn",
        id: Optional[str] = None,
        use_canonize: bool = True,
        use_lattice: bool = False,
        use_angle: bool = False,
    ):
        # print('id',id)
        if neighbor_strategy == "k-nearest":
            edges, a, b, c = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            u, v, r, l, nei, atom_lat = build_undirected_edgedata(atoms, edges, a, b, c)
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features="atomic_number"))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        atom_lat = atom_lat.repeat(node_features.shape[0],1,1)
        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()

        g = Data(x=node_features, edge_index=edge_index, edge_attr=r, edge_type=l, edge_nei=nei, atom_lat=atom_lat)

        features = get_attribute_lookup(atom_features)
        z = g.x
        g.atomic_number = z
        z = z.type(torch.IntTensor).squeeze()
        f = torch.tensor(features[z]).type(torch.FloatTensor)
        if g.x.size(0) == 1:
            f = f.unsqueeze(0)
        g.x = f

        return g

class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        graphs: Sequence[Data],
        targets=None,
        transform=None,
        line_graph=False,
        classification=False,
        ids=None,
    ):
        """Pytorch Dataset for atomistic graphs.

        """
        self.graphs = graphs
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
            return g, g, label, cif_id

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
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        return (batched_graph, batched_line_graph, batched_line_graph), torch.stack(labels, dim=0), batch_cif_ids


def load_comformer_data(root_dir: str='data/', task: str=None, config: dict=None):
    assert os.path.exists(root_dir), 'root_dir does not exist!'
    id_prop_file = os.path.join(root_dir, task, 'targets.csv')
    assert os.path.exists(id_prop_file), 'targets.csv does not exist!'
    if os.path.exists(os.path.join(root_dir, task, "comformer_data.pkl")) == True:
        with open(os.path.join(root_dir, task, "comformer_data.pkl"), 'rb') as f:
            data = pickle.load(f)
    else:
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            id_prop_data = [row for row in reader]
        data = []
        for d in id_prop_data:
            cif_id, target = d
            try:
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
                    use_canonize=True,
                    use_lattice=True,
                    use_angle=False,
                    id=cif_id,
                )
                data.append([g, torch.Tensor([float(target)]), cif_id])
            except Exception as e:
                print(f"Failed at {cif_id}.cif: {e}")
        with open(os.path.join(root_dir, task, "comformer_data.pkl"), 'wb') as f:
            pickle.dump(data, f)

    return data

def get_comformer_train_val_test_loader(dataset, train_indexs=None,
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


    train_dataset = [dataset[x][0] for x in train_indice]
    train_targets = [dataset[x][1] for x in train_indice]
    train_ids = [dataset[x][2] for x in train_indice]

    val_dataset = [dataset[x][0] for x in val_indice]
    val_targets = [dataset[x][1] for x in val_indice]
    val_ids = [dataset[x][2] for x in val_indice]

    test_dataset = [dataset[x][0] for x in test_indice]
    test_targets = [dataset[x][1] for x in test_indice]
    test_ids = [dataset[x][2] for x in test_indice]

    train_data = PygStructureDataset(
        train_dataset,
        targets=train_targets,
        line_graph=True,
        ids=train_ids,
    )
    val_data = PygStructureDataset(
        val_dataset,
        targets=val_targets,
        line_graph=True,
        ids=val_ids,
    )
    test_data = PygStructureDataset(
        test_dataset,
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

class ComformerConv(MessagePassing):
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
        super(ComformerConv, self).__init__(node_dim=0, **kwargs)

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
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)

        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                            nn.SiLU(),
                                            nn.Linear(out_channels, out_channels))
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.bn = nn.BatchNorm1d(out_channels)
        self.bn_att = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        print('I am using the correct version of matformer')

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

        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_concate(out)

        return self.softplus(x[1] + self.bn(out))

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key_j = self.key_update(torch.cat((key_i, key_j, edge_attr), dim=-1))
        alpha = (query_i * key_j) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1))
        out = out * self.sigmoid(self.bn_att(alpha.view(-1, self.out_channels)).view(-1, self.heads, self.out_channels))
        return out


class ComformerConv_edge(nn.Module):
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
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lemb = nn.Embedding(num_embeddings=3, embedding_dim=32)
        self.embedding_dim = 32
        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        # for test
        self.lin_key_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e3 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e3 = nn.Linear(in_channels[0], heads * out_channels)
        # for test ends
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge_len = nn.Linear(in_channels[0] + self.embedding_dim, in_channels[0])
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                            nn.SiLU(),
                                            nn.Linear(out_channels, out_channels))
        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.bn_att = nn.BatchNorm1d(out_channels)

        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        print('I am using the invariant version of EPCNet')

    def forward(self, edge: Union[Tensor, PairTensor], edge_nei_len: OptTensor = None,
                edge_nei_angle: OptTensor = None):
        # preprocess for edge of shape [num_edges, hidden_dim]

        H, C = self.heads, self.out_channels
        if isinstance(edge, Tensor):
            edge: PairTensor = (edge, edge)
        device = edge[1].device
        query_x = self.lin_query(edge[1]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        key_x = self.lin_key(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        value_x = self.lin_value(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        num_edge = query_x.shape[0]

        key_y = torch.cat((self.lin_key_e1(edge_nei_len[:, 0, :]).view(-1, 1, H, C),
                           self.lin_key_e2(edge_nei_len[:, 1, :]).view(-1, 1, H, C),
                           self.lin_key_e3(edge_nei_len[:, 2, :]).view(-1, 1, H, C)), dim=1)
        value_y = torch.cat((self.lin_value_e1(edge_nei_len[:, 0, :]).view(-1, 1, H, C),
                             self.lin_value_e2(edge_nei_len[:, 1, :]).view(-1, 1, H, C),
                             self.lin_value_e3(edge_nei_len[:, 2, :]).view(-1, 1, H, C)), dim=1)

        # preprocess for interaction of shape [num_edges, 3, hidden_dim]
        edge_xy = self.lin_edge(edge_nei_angle).view(-1, 3, H, C)

        key = self.key_update(torch.cat((key_x, key_y, edge_xy), dim=-1))
        alpha = (query_x * key) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_x, value_y, edge_xy), dim=-1))
        out = out * self.sigmoid(
            self.bn_att(alpha.view(-1, self.out_channels)).view(-1, 3, self.heads, self.out_channels))

        out = out.view(-1, 3, self.heads * self.out_channels)
        out = self.lin_concate(out)
        # aggregate the msg
        out = out.sum(dim=1)

        return self.softplus(edge[1] + self.bn(out))


class TensorProductConvLayer(torch.nn.Module):
    # from Torsional diffusion
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.Softplus(),
            nn.Linear(n_edge_features, tp.weight_numel)
        )

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        return out


class ComformerConvEqui(nn.Module):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            edge_dim: Optional[int] = None,
            use_second_order_repr: bool = True,
            ns: int = 64,
            nv: int = 8,
            residual: bool = True,
    ):
        super().__init__()

        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o + {nv}x2e',
            f'{ns}x0e'
        ]
        self.ns, self.nv = ns, nv
        self.node_linear = nn.Linear(in_channels, ns)
        self.skip_linear = nn.Linear(in_channels, out_channels)
        self.sh = '1x0e + 1x1o + 1x2e'
        self.nlayer_1 = TensorProductConvLayer(
            in_irreps=irrep_seq[0],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual
        )
        self.nlayer_2 = TensorProductConvLayer(
            in_irreps=irrep_seq[1],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[2],
            n_edge_features=edge_dim,
            residual=False
        )
        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(ns)
        self.node_linear_2 = nn.Linear(ns, out_channels)

    def forward(self, data, node_feature: Union[Tensor, PairTensor], edge_index: Adj,
                edge_feature: Union[Tensor, PairTensor],
                edge_nei_len: OptTensor = None):
        edge_vec = data.edge_attr
        edge_irr = o3.spherical_harmonics(self.sh, edge_vec, normalize=True, normalization='component')
        n_ = node_feature.shape[0]
        skip_connect = node_feature
        node_feature = self.node_linear(node_feature)
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.nlayer_2(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.softplus(self.node_linear_2(self.softplus(self.bn(node_feature))))
        node_feature += self.skip_linear(skip_connect)

        return node_feature


def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
            torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

class eComFormer(nn.Module): # eComFormer
    """att pyg implementation."""

    def __init__(
        self,
        conv_layers: int = 3,
        atom_input_features: int = 92,
        edge_features: int = 256,
        node_features: int = 256,
        fc_features: int = 256,
        output_features: int = 1,
        node_layer_head: int = 1,
        evidential: str = "False",
        classification: bool = False,
    ):
        """Set up att modules."""
        super().__init__()
        self.classification = classification
        self.evidential = evidential
        self.atom_embedding = nn.Linear(
            atom_input_features, node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=edge_features,
            ),
            nn.Linear(edge_features, node_features),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=node_features, out_channels=node_features, heads=node_layer_head, edge_dim=node_features)
                for _ in range(conv_layers)
            ]
        )

        self.equi_update = ComformerConvEqui(in_channels=node_features, out_channels=node_features, edge_dim=node_features, use_second_order_repr=True)

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
        data, _, _ = data
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        edge_features = self.rbf(edge_feat)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.equi_update(data, node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")


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


class iComFormer(nn.Module): # iComFormer
    """att pyg implementation."""

    def __init__(
        self,
        conv_layers: int = 3,
        atom_input_features: int = 92,
        edge_features: int = 256,
        triplet_input_features: int =  256,
        node_features: int = 256,
        fc_features: int = 256,
        output_features: int = 1,
        node_layer_head: int = 1,
        evidential: str = "False",
        classification: bool = False,
    ):
        """Set up att modules."""
        super().__init__()
        self.classification = classification
        self.evidential = evidential
        self.atom_embedding = nn.Linear(
            atom_input_features, node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=edge_features,
            ),
            nn.Linear(edge_features, node_features),
            nn.Softplus(),
        )

        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=triplet_input_features,
            ),
            nn.Linear(triplet_input_features, node_features),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=node_features, out_channels=node_features, heads=node_layer_head, edge_dim=node_features)
                for _ in range(conv_layers)
            ]
        )

        self.edge_update_layer = ComformerConv_edge(in_channels=node_features, out_channels=node_features, heads=node_layer_head, edge_dim=node_features)

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
        data, _, _ = data
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1) # [num_edges]
        edge_nei_len = -0.75 / torch.norm(data.edge_nei, dim=-1) # [num_edges, 3]
        edge_nei_angle = bond_cosine(data.edge_nei, data.edge_attr.unsqueeze(1).repeat(1, 3, 1)) # [num_edges, 3, 3] -> [num_edges, 3]
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat)
        edge_nei_len = self.rbf(edge_nei_len.reshape(-1)).reshape(num_edge, 3, -1)
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape(-1)).reshape(num_edge, 3, -1)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        edge_features = self.edge_update_layer(edge_features, edge_nei_len, edge_nei_angle)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

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



