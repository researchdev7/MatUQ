'''
import os
import csv
from pymatgen.core.structure import Structure
import tensorflow as tf
import keras as ks
from keras.layers import Dropout
from kgcnn.layers.scale import get as get_scaler
from kgcnn.layers.modules import Input
from keras.layers import Add, Subtract, Concatenate, Dense, Multiply, Layer
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, BesselBasisLayer, EdgeAngle, ShiftPeriodicLattice, \
    SphericalBasisLayer
from kgcnn.layers.gather import GatherNodes, GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use
from keras import ops
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.update import ResidualLayer
from kgcnn.initializers.initializers import GlorotOrthogonal, HeOrthogonal

def load_dimenetpp_data(root_dir: str='data/', task: str=None):
    assert os.path.exists(root_dir), 'root_dir does not exist!'
    id_prop_file = os.path.join(root_dir, task, 'targets.csv')
    assert os.path.exists(id_prop_file), 'targets.csv does not exist!'
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]
    cif_ids = []
    targets = []
    crystals = []
    for d in id_prop_data:
        cif_id, target = d
        cif_ids.append(cif_id)
        targets.append(float(target))
        crystal = Structure.from_file(os.path.join(root_dir, task,
                                                   cif_id + '.cif'))
        crystals.append(crystal)
    return cif_ids, targets, crystals

class DimNetInteractionPPBlock(Layer):
    """DimNetPP Interaction Block as defined by `DimNetPP <https://arxiv.org/abs/2011.14115>`__ .

    Args:
        emb_size: Embedding size used for the messages
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size: Embedding size used inside the basis transformation
        num_before_skip: Number of residual layers in interaction block before skip connection
        num_after_skip: Number of residual layers in interaction block before skip connection
        use_bias (bool, optional): Use bias. Defaults to True.
        pooling_method (str): Pooling method information for layer. Default is 'sum'.
        activation (str): Activation function. Default is "swish".
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'kgcnn>glorot_orthogonal'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self, emb_size,
                 int_emb_size,
                 basis_emb_size,
                 num_before_skip,
                 num_after_skip,
                 use_bias=True,
                 pooling_method="sum",
                 activation='swish',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer="kgcnn>glorot_orthogonal",  # default is 'kgcnn>glorot_orthogonal'
                 bias_initializer='zeros',
                 **kwargs):
        super(DimNetInteractionPPBlock, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.pooling_method = pooling_method
        self.emb_size = emb_size
        self.int_emb_size = int_emb_size
        self.basis_emb_size = basis_emb_size
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = Dense(basis_emb_size, use_bias=False, **kernel_args)
        self.dense_rbf2 = Dense(emb_size, use_bias=False, **kernel_args)
        self.dense_sbf1 = Dense(basis_emb_size, use_bias=False, **kernel_args)
        self.dense_sbf2 = Dense(int_emb_size, use_bias=False, **kernel_args)

        # Dense transformations of input messages
        self.dense_ji = Dense(emb_size, activation=activation, use_bias=True, **kernel_args)
        self.dense_kj = Dense(emb_size, activation=activation, use_bias=True, **kernel_args)

        # Embedding projections for interaction triplets
        self.down_projection = Dense(int_emb_size, activation=activation, use_bias=False, **kernel_args)
        self.up_projection = Dense(emb_size, activation=activation, use_bias=False, **kernel_args)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args))
        self.final_before_skip = Dense(emb_size, activation=activation, use_bias=True, **kernel_args)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args))

        self.lay_add1 = Add()
        self.lay_add2 = Add()
        self.lay_mult1 = Multiply()
        self.lay_mult2 = Multiply()

        self.lay_gather = GatherNodesOutgoing()  # Are edges here
        self.lay_pool = AggregateLocalEdges(pooling_method=pooling_method)

    def build(self, input_shape):
        """Build layer."""
        super(DimNetInteractionPPBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [edges, rbf, sbf, angle_index]

                - edges (Tensor): Edge embeddings of shape ([M], F)
                - rbf (Tensor): Radial basis features of shape ([M], F)
                - sbf (Tensor): Spherical basis features of shape ([K], F)
                - angle_index (Tensor): Angle indices referring to two edges of shape (2, [K])

        Returns:
            Tensor: Updated edge embeddings.
        """
        x, rbf, sbf, id_expand = inputs

        # Initial transformation
        x_ji = self.dense_ji(x, **kwargs)
        x_kj = self.dense_kj(x, **kwargs)

        # Transform via Bessel basis
        rbf = self.dense_rbf1(rbf, **kwargs)
        rbf = self.dense_rbf2(rbf, **kwargs)
        x_kj = self.lay_mult1([x_kj, rbf], **kwargs)

        # Down-project embeddings and generate interaction triplet embeddings
        x_kj = self.down_projection(x_kj, **kwargs)
        x_kj = self.lay_gather([x_kj, id_expand], **kwargs)

        # Transform via 2D spherical basis
        sbf = self.dense_sbf1(sbf, **kwargs)
        sbf = self.dense_sbf2(sbf, **kwargs)
        x_kj = self.lay_mult2([x_kj, sbf], **kwargs)

        # Aggregate interactions and up-project embeddings
        x_kj = self.lay_pool([rbf, x_kj, id_expand], **kwargs)
        x_kj = self.up_projection(x_kj, **kwargs)

        # Transformations before skip connection
        x2 = self.lay_add1([x_ji, x_kj], **kwargs)
        for layer in self.layers_before_skip:
            x2 = layer(x2, **kwargs)
        x2 = self.final_before_skip(x2, **kwargs)

        # Skip connection
        x = self.lay_add2([x, x2],**kwargs)

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            x = layer(x, **kwargs)

        return x

    def get_config(self):
        config = super(DimNetInteractionPPBlock, self).get_config()
        config.update({"use_bias": self.use_bias, "pooling_method": self.pooling_method, "emb_size": self.emb_size,
                       "int_emb_size": self.int_emb_size, "basis_emb_size": self.basis_emb_size,
                       "num_before_skip": self.num_before_skip, "num_after_skip": self.num_after_skip})
        conf_dense = self.dense_ji.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            if x in conf_dense:
                config.update({x: conf_dense[x]})
        return config


class DimNetOutputBlock(Layer):
    """DimNetPP Output Block as defined by `DimNetPP <https://arxiv.org/abs/2011.14115>`__ .

    Args:
        emb_size (list): List of node embedding dimension.
        out_emb_size (list): List of edge embedding dimension.
        num_dense (list): Number of dense layer for MLP.
        num_targets (int): Number of output target dimension. Defaults to 12.
        use_bias (bool, optional): Use bias. Defaults to True.
        kernel_initializer: Initializer for kernels. Default is 'glorot_orthogonal' with fallback 'orthogonal'.
        output_kernel_initializer: Initializer for last kernel. Default is 'zeros'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        activation (str): Activation function. Default is 'kgcnn>swish'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        pooling_method (str): Pooling method information for layer. Default is 'mean'.
    """

    def __init__(self, emb_size,
                 out_emb_size,
                 num_dense,
                 num_targets=12,
                 use_bias=True,
                 output_kernel_initializer="zeros", kernel_initializer='kgcnn>glorot_orthogonal',
                 bias_initializer='zeros',
                 activation='swish',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 pooling_method="sum",
                 **kwargs):
        """Initialize layer."""
        super(DimNetOutputBlock, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.emb_size = emb_size
        self.out_emb_size = out_emb_size
        self.num_dense = num_dense
        self.num_targets = num_targets
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_initializer": bias_initializer,
                       "bias_regularizer": bias_regularizer, "bias_constraint": bias_constraint, }

        self.dense_rbf = Dense(emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.up_projection = Dense(out_emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.dense_mlp = GraphMLP([out_emb_size] * num_dense, activation=activation,
                                  kernel_initializer=kernel_initializer, use_bias=use_bias, **kernel_args)
        self.dimnet_mult = Multiply()
        self.pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.dense_final = Dense(num_targets, use_bias=False, kernel_initializer=output_kernel_initializer,
                                 **kernel_args)
        self.dropout = Dropout(rate=0.1)

    def build(self, input_shape):
        """Build layer."""
        super(DimNetOutputBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, tensor_index, state]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - rbf (Tensor): Edge distance basis of shape ([M], F)
                - tensor_index (Tensor): Edge indices referring to nodes of shape (2, [M])

        Returns:
            Tensor: Updated node embeddings of shape ([N], F_T).
        """
        # Calculate edge Update
        n_atoms, x, rbf, idnb_i = inputs
        g = self.dense_rbf(rbf, **kwargs)
        x = self.dimnet_mult([g, x], **kwargs)
        x = self.pool([n_atoms, x, idnb_i], **kwargs)
        x = self.up_projection(x, **kwargs)
        x = self.dense_mlp(x, **kwargs)
        x = self.dropout(x, training=kwargs.get('training', False))
        x = self.dense_final(x, **kwargs)
        return x

    def get_config(self):
        config = super(DimNetOutputBlock, self).get_config()
        conf_mlp = self.dense_mlp.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            if x in conf_mlp:
                config.update({x: conf_mlp[x][0]})
        conf_dense_output = self.dense_final.get_config()
        config.update({"output_kernel_initializer": conf_dense_output["kernel_initializer"]})
        config.update({"pooling_method": self.pooling_method, "use_bias": self.use_bias})
        config.update({"emb_size": self.emb_size, "out_emb_size": self.out_emb_size, "num_dense": self.num_dense,
                       "num_targets": self.num_targets})
        return config


class EmbeddingDimeBlock(Layer):
    """Custom Embedding Block of `DimNetPP <https://arxiv.org/abs/2011.14115>`__ .

    Naming of inputs here should match keras Embedding layer.

    Args:
        input_dim (int): Integer. Size of the vocabulary, i.e. maximum integer index + 1.
        output_dim (int): Integer. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the embeddings matrix (see keras.initializers).
        embeddings_regularizer: Regularizer function applied to the embeddings matrix (see keras.regularizers).
        embeddings_constraint: Constraint function applied to the embeddings matrix (see keras.constraints).

    """
    def __init__(self,
                 input_dim,  # Vocabulary
                 output_dim,  # Embedding size
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        super(EmbeddingDimeBlock, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.embeddings_initializer = ks.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = ks.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = ks.constraints.get(embeddings_constraint)

        # Original implementation used initializer:
        # embeddings_initializer = {'class_name': 'RandomUniform', 'config': {'minval': -1.7320508075688772,
        # 'maxval': 1.7320508075688772, 'seed': None}}
        self.embeddings = self.add_weight(name="embeddings", shape=(self.input_dim + 1, self.output_dim),
                                          dtype=self.dtype, initializer=self.embeddings_initializer,
                                          regularizer=self.embeddings_regularizer,
                                          constraint=self.embeddings_constraint,
                                          trainable=True)

    def call(self, inputs, **kwargs):
        """Embedding of inputs. Forward pass."""
        out = ops.take(self.embeddings, tf.cast(inputs, dtype=tf.int32), axis=0)
        return out

    def get_config(self):
        config = super(EmbeddingDimeBlock, self).get_config()
        config.update({"input_dim": self.input_dim, "output_dim": self.output_dim,
                       "embeddings_initializer": ks.initializers.serialize(self.embeddings_initializer),
                       "embeddings_regularizer": ks.regularizers.serialize(self.embeddings_regularizer),
                       "embeddings_constraint": ks.constraints.serialize(self.embeddings_constraint)
                       })
        return config

class OutputProcessingLayer(Layer):
    def __init__(self, num_splits=4, min_val=1e-6, **kwargs):
        super(OutputProcessingLayer, self).__init__(**kwargs)
        self.num_splits = num_splits  # Number of output components (mu, logv, logalpha, logbeta)
        self.min_val = min_val  # Minimum value for stability

    def call(self, inputs, **kwargs):
        out = inputs

        # During eager execution, apply shape logic if needed
        if tf.executing_eagerly():
            if tf.shape(out)[0] == 4:
                out = tf.expand_dims(out, 0)
            out = tf.reshape(out, [tf.shape(out)[0], -1, self.num_splits])

        # Split into components
        mu, logv, logalpha, logbeta = tf.split(out, num_or_size_splits=self.num_splits, axis=-1)

        # Squeeze and apply transformations
        mu = tf.squeeze(mu, axis=-1)
        logv = tf.squeeze(logv, axis=-1)
        logalpha = tf.squeeze(logalpha, axis=-1)
        logbeta = tf.squeeze(logbeta, axis=-1)

        # Apply softplus and minimum value for stability
        return (
            mu,
            tf.nn.softplus(logv) + self.min_val,
            tf.nn.softplus(logalpha) + self.min_val + 1,
            tf.nn.softplus(logbeta) + self.min_val
        )

    def compute_output_shape(self, input_shape):
        # Input shape is (batch_size, features)
        # Output is a tuple of 4 tensors, each (batch_size, num_class)
        batch_size, features = input_shape
        if features is not None and features % self.num_splits != 0:
            raise ValueError(f"Input feature dimension {features} must be divisible by {self.num_splits}")
        output_shape = (batch_size,)
        return tuple([output_shape] * self.num_splits)

    def get_config(self):
        config = super(OutputProcessingLayer, self).get_config()
        config.update({"num_splits": self.num_splits, "min_val": self.min_val})
        return config

def model_disjoint(
        inputs,
        use_node_embedding,
        input_node_embedding: dict = None,
        emb_size: int = None,
        out_emb_size: int = None,
        int_emb_size: int = None,
        basis_emb_size: int = None,
        num_blocks: int = None,
        num_spherical: int = None,
        num_radial: int = None,
        cutoff: float = None,
        envelope_exponent: int = None,
        num_before_skip: int = None,
        num_after_skip: int = None,
        num_dense_output: int = None,
        num_targets: int = None,
        activation: str = None,
        extensive: bool = None,
        output_init: str = None,
        use_output_mlp: bool = None,
        output_embedding: str = None,
        output_mlp: dict = None
):
    n, x, edi, adi, batch_id_node, count_nodes = inputs

    # Atom embedding
    if use_node_embedding:
        n = EmbeddingDimeBlock(**input_node_embedding)(n)

    # Calculate distances
    pos1, pos2 = NodePosition()([x, edi])
    d = NodeDistanceEuclidean()([pos1, pos2])
    rbf = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)(d)

    # Calculate angles
    v12 = Subtract()([pos1, pos2])
    a = EdgeAngle()([v12, adi])
    sbf = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                              envelope_exponent=envelope_exponent)([d, a, adi])

    # Embedding block
    rbf_emb = Dense(emb_size, use_bias=True, activation=activation,
                    kernel_initializer="kgcnn>glorot_orthogonal")(rbf)
    n_pairs = GatherNodes()([n, edi])
    x = Concatenate(axis=-1)([n_pairs, rbf_emb])
    x = Dense(emb_size, use_bias=True, activation=activation, kernel_initializer="kgcnn>glorot_orthogonal")(x)
    ps = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                           output_kernel_initializer=output_init)([n, x, rbf, edi])

    # Interaction blocks
    add_xp = Add()
    for i in range(num_blocks):
        x = DimNetInteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip)(
            [x, rbf, sbf, adi])

        p_update = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                                     output_kernel_initializer=output_init)([n, x, rbf, edi])
        ps = add_xp([ps, p_update])

    if extensive:
        out = PoolingNodes(pooling_method="sum")([count_nodes, ps, batch_id_node])
    else:
        out = PoolingNodes(pooling_method="mean")([count_nodes, ps, batch_id_node])

    if use_output_mlp:
        out = MLP(**output_mlp)(out)

    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `DimeNetPP`. ")

    if tf.shape(out)[0] == 4:
        out = tf.expand_dims(out, 0)
    out = tf.reshape(out, [tf.shape(out)[0], -1, 4])
    mu, logv, logalpha, logbeta = tf.split(out, num_or_size_splits=4, axis=-1)
    mu = tf.squeeze(mu, axis=-1)
    logv = tf.squeeze(logv, axis=-1)
    logalpha = tf.squeeze(logalpha, axis=-1)
    logbeta = tf.squeeze(logbeta, axis=-1)

    min_val = 1e-6
    return (mu,
            tf.nn.softplus(logv) + min_val,
            tf.nn.softplus(logalpha) + min_val + 1,
            tf.nn.softplus(logbeta) + min_val)


def model_disjoint_crystal(
        inputs,
        use_node_embedding,
        input_node_embedding: dict = None,
        emb_size: int = None,
        out_emb_size: int = None,
        int_emb_size: int = None,
        basis_emb_size: int = None,
        num_blocks: int = None,
        num_spherical: int = None,
        num_radial: int = None,
        cutoff: float = None,
        envelope_exponent: int = None,
        num_before_skip: int = None,
        num_after_skip: int = None,
        num_dense_output: int = None,
        num_targets: int = None,
        activation: str = None,
        extensive: bool = None,
        output_init: str = None,
        use_output_mlp: bool = None,
        output_embedding: str = None,
        output_mlp: dict = None
    ):

    n, x, edi, adi, edge_image, lattice, batch_id_node, batch_id_edge, count_nodes = inputs

    # Atom embedding
    if use_node_embedding:
        n = EmbeddingDimeBlock(**input_node_embedding)(n)

    # Calculate distances
    pos1, pos2 = NodePosition()([x, edi])
    pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice, batch_id_edge])
    d = NodeDistanceEuclidean()([pos1, pos2])
    rbf = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)(d)

    # Calculate angles
    v12 = Subtract()([pos1, pos2])
    a = EdgeAngle()([v12, adi])
    sbf = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                              envelope_exponent=envelope_exponent)([d, a, adi])

    # Embedding block
    rbf_emb = Dense(emb_size, use_bias=True, activation=activation,
                    kernel_initializer="kgcnn>glorot_orthogonal")(rbf)
    n_pairs = GatherNodes()([n, edi])
    x = Concatenate(axis=-1)([n_pairs, rbf_emb])
    x = Dense(emb_size, use_bias=True, activation=activation, kernel_initializer="kgcnn>glorot_orthogonal")(x)
    ps = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=4,
                           output_kernel_initializer=output_init)([n, x, rbf, edi])

    # Interaction blocks
    add_xp = Add()
    for i in range(num_blocks):
        x = DimNetInteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip)(
            [x, rbf, sbf, adi])
        p_update = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=4,
                                     output_kernel_initializer=output_init)([n, x, rbf, edi])
        ps = add_xp([ps, p_update])

    if extensive:
        out = PoolingNodes(pooling_method="sum")([count_nodes, ps, batch_id_node])
    else:
        out = PoolingNodes(pooling_method="mean")([count_nodes, ps, batch_id_node])

    if use_output_mlp:
        out = MLP(**output_mlp)(out)

    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `DimeNetPP`. ")

    out = OutputProcessingLayer(num_splits=4, min_val=1e-6)(out)

    return out

# To be updated if model is changed in a significant way.
__model_version__ = "2023-12-04"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'DimeNetPP' is not supported." % backend_to_use())

# Implementation of DimeNet++ in `keras` from paper:
# Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules
# Johannes Klicpera, Shankari Giri, Johannes T. Margraf, Stephan Günnemann
# https://arxiv.org/abs/2011.14115
# Original code: https://github.com/gasteigerjo/dimenet

model_default = {
    "name": "DimeNetPP",
    "inputs": [
        {"shape": [None], "name": "node_number", "dtype": "int64"},
        {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
        {"shape": [None, 2], "name": "angle_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"},
        {"shape": (), "name": "total_angles", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {
        "input_dim": 95, "output_dim": 128, "embeddings_initializer": {
            "class_name": "RandomUniform",
            "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}}
    },
    "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
    "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
    "cutoff": 5.0, "envelope_exponent": 5,
    "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
    "num_targets": 64, "extensive": True, "output_init": "zeros",
    "activation": "swish", "verbose": 10,
    "output_embedding": "graph",
    "use_output_mlp": True,
    "output_tensor_type": "padded",
    "output_scaling": None,
    "output_mlp": {"use_bias": [True, False],
                   "units": [64, 12], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               emb_size: int = None,
               out_emb_size: int = None,
               int_emb_size: int = None,
               basis_emb_size: int = None,
               num_blocks: int = None,
               num_spherical: int = None,
               num_radial: int = None,
               cutoff: float = None,
               envelope_exponent: int = None,
               num_before_skip: int = None,
               num_after_skip: int = None,
               num_dense_output: int = None,
               num_targets: int = None,
               activation: str = None,
               extensive: bool = None,
               output_init: str = None,
               verbose: int = None,
               name: str = None,
               output_embedding: str = None,
               output_tensor_type: str = None,
               use_output_mlp: bool = None,
               output_mlp: dict = None,
               output_scaling: dict = None
               ):
    """Make `DimeNetPP <https://arxiv.org/abs/2011.14115>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DimeNetPP.model_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, angle_indices...]`
    with '...' indicating mask or ID tensors following the template below.
    Note that you must supply angle indices as index pairs that refer to two edges.

    %s

    **Model outputs**:
    The standard output template:

    %s


    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        emb_size (int): Overall embedding size used for the messages.
        out_emb_size (int): Embedding size for output of :obj:`DimNetOutputBlock`.
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used inside the basis transformation.
        num_blocks (int): Number of graph embedding blocks or depth of the network.
        num_spherical (int): Number of spherical components in :obj:`SphericalBasisLayer`.
        num_radial (int): Number of radial components in basis layer.
        cutoff (float): Distance cutoff for basis layer.
        envelope_exponent (int): Exponent in envelope function for basis layer.
        num_before_skip (int): Number of residual layers in interaction block before skip connection
        num_after_skip (int): Number of residual layers in interaction block after skip connection
        num_dense_output (int): Number of dense units in output :obj:`DimNetOutputBlock`.
        num_targets (int): Number of targets or output embedding dimension of the model.
        activation (str, dict): Activation to use.
        extensive (bool): Graph output for extensive target to apply sum for pooling or mean otherwise.
        output_init (str, dict): Output initializer for kernel.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final :obj:`MLP`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation. Note that DimeNetPP originally defines the output dimension
            via `num_targets`. But this can be set to `out_emb_size` and the `output_mlp` be used for more
            specific control.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    dj = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0, 1, 2],
        index_assignment=[None, None, 0, 2]
    )

    n, x, edi, adi, batch_id_node, batch_id_edge, batch_id_angles, node_id, edge_id, angle_id, count_nodes, count_edges, count_angles = dj

    out = model_disjoint(
        [n, x, edi, adi, batch_id_node, count_nodes],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        emb_size=emb_size,
        out_emb_size=out_emb_size,
        int_emb_size=int_emb_size,
        basis_emb_size=basis_emb_size,
        num_blocks=num_blocks,
        num_spherical=num_spherical,
        num_radial=num_radial,
        cutoff=cutoff,
        envelope_exponent=envelope_exponent,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_dense_output=num_dense_output,
        num_targets=num_targets,
        activation=activation,
        extensive=extensive,
        output_init=output_init,
        use_output_mlp=use_output_mlp,
        output_embedding=output_embedding,
        output_mlp=output_mlp
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out = scaler([out, n, batch_id_node])
        else:
            out = scaler(out)

    # Output embedding choice
    out = template_cast_output(
        [out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs,
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_model.__doc__ = make_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)

model_crystal_default = {
    "name": "DimeNetPP",
    "inputs": [
        {"shape": [None], "name": "node_number", "dtype": "int64", "ragged": True},
        {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
        {'shape': (None, 3), 'name': "edge_image", 'dtype': 'int64', 'ragged': True},
        {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
    ],
    "input_tensor_type": "ragged",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {
        "input_dim": 95, "output_dim": 128, "embeddings_initializer": {
            "class_name": "RandomUniform",
            "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}}
    },
    "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
    "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
    "cutoff": 5.0, "envelope_exponent": 5,
    "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
    "num_targets": 64, "extensive": True, "output_init": "zeros",
    "activation": "swish", "verbose": 10,
    "output_embedding": "graph",
    "use_output_mlp": True,
    "output_tensor_type": "padded",
    "output_scaling": None,
    "output_mlp": {"use_bias": [True, False],
                   "units": [64, 12], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_crystal_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_crystal_model(inputs: list = None,
                       input_tensor_type: str = None,
                       cast_disjoint_kwargs: dict = None,
                       input_embedding: dict = None,
                       input_node_embedding: dict = None,
                       emb_size: int = None,
                       out_emb_size: int = None,
                       int_emb_size: int = None,
                       basis_emb_size: int = None,
                       num_blocks: int = None,
                       num_spherical: int = None,
                       num_radial: int = None,
                       cutoff: float = None,
                       envelope_exponent: int = None,
                       num_before_skip: int = None,
                       num_after_skip: int = None,
                       num_dense_output: int = None,
                       num_targets: int = None,
                       activation: str = None,
                       extensive: bool = None,
                       output_init: str = None,
                       verbose: int = None,
                       name: str = None,
                       output_embedding: str = None,
                       output_tensor_type: str = None,
                       use_output_mlp: bool = None,
                       output_mlp: dict = None,
                       output_scaling: dict = None
                       ):
    """Make `DimeNetPP <https://arxiv.org/abs/2011.14115>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DimeNetPP.model_crystal_default`.

    .. note::

        DimeNetPP does require a large amount of memory for this implementation, which increase quickly with
        the number of connections in a batch. Use ragged input or dataloader if possible.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, angle_indices, image_translation, lattice, ...]`
    with '...' indicating mask or ID tensors following the template below.
    Note that you must supply angle indices as index pairs that refer to two edges.

    %s

    **Model outputs**:
    The standard output template:

    %s


    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        emb_size (int): Overall embedding size used for the messages.
        out_emb_size (int): Embedding size for output of :obj:`DimNetOutputBlock`.
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used inside the basis transformation.
        num_blocks (int): Number of graph embedding blocks or depth of the network.
        num_spherical (int): Number of spherical components in :obj:`SphericalBasisLayer`.
        num_radial (int): Number of radial components in basis layer.
        cutoff (float): Distance cutoff for basis layer.
        envelope_exponent (int): Exponent in envelope function for basis layer.
        num_before_skip (int): Number of residual layers in interaction block before skip connection
        num_after_skip (int): Number of residual layers in interaction block after skip connection
        num_dense_output (int): Number of dense units in output :obj:`DimNetOutputBlock`.
        num_targets (int): Number of targets or output embedding dimension of the model.
        activation (str, dict): Activation to use.
        extensive (bool): Graph output for extensive target to apply sum for pooling or mean otherwise.
        output_init (str, dict): Output initializer for kernel.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final :obj:`MLP`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation. Note that DimeNetPP originally defines the output dimension
            via `num_targets`. But this can be set to `out_emb_size` and the `output_mlp` be used for more
            specific control.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    disjoint_inputs = template_cast_list_input(
        model_inputs, input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        index_assignment=[None, None, 0, 2, None, None],
        mask_assignment=[0, 0, 1, 2, 1, None]
    )
    n, x, edi, angi, img, lattice, batch_id_node, batch_id_edge, batch_id_angles, node_id, edge_id, angle_id, count_nodes, count_edges, count_angles = disjoint_inputs

    # Wrap disjoint model
    out = model_disjoint_crystal(
        [n, x, edi, angi, img, lattice, batch_id_node, batch_id_edge, count_nodes],
        use_node_embedding=(len(inputs[0]["shape"]) == 1) if input_node_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        emb_size=emb_size,
        out_emb_size=out_emb_size,
        int_emb_size=int_emb_size,
        basis_emb_size=basis_emb_size,
        num_blocks=num_blocks,
        num_spherical=num_spherical,
        num_radial=num_radial,
        cutoff=cutoff,
        envelope_exponent=envelope_exponent,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_dense_output=num_dense_output,
        num_targets=num_targets,
        activation=activation,
        extensive=extensive,
        output_init=output_init,
        use_output_mlp=use_output_mlp,
        output_embedding=output_embedding,
        output_mlp=output_mlp
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out = scaler([out, n, batch_id_node])
        else:
            out = scaler(out)

    # Output embedding choice
    out = template_cast_output(
        [out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs,
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_crystal_model.__doc__ = make_crystal_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
'''

import os
import csv
from pymatgen.core.structure import Structure
import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.conv.dimenet_conv import DimNetInteractionPPBlock, EmbeddingDimeBlock, SphericalBasisLayer
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.geom import NodeDistanceEuclidean, EdgeAngle, BesselBasisLayer, NodePosition, ShiftPeriodicLattice
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate, LazyAdd, LazySubtract, LazyMultiply
from kgcnn.layers.pooling import PoolingNodes, PoolingLocalEdges
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.mlp import MLP, GraphMLP

ks = tf.keras

def load_dimenetpp_data(root_dir: str='data/', task: str=None):
    assert os.path.exists(root_dir), 'root_dir does not exist!'
    id_prop_file = os.path.join(root_dir, task, 'targets.csv')
    assert os.path.exists(id_prop_file), 'targets.csv does not exist!'
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]
    cif_ids = []
    targets = []
    crystals = []
    for d in id_prop_data:
        cif_id, target = d
        cif_ids.append(cif_id)
        targets.append(float(target))
        crystal = Structure.from_file(os.path.join(root_dir, task,
                                                   cif_id + '.cif'))
        crystals.append(crystal)
    return cif_ids, targets, crystals

@ks.utils.register_keras_serializable(package='DimeNetPP', name='DimNetOutputBlock')
class DimNetOutputBlock(GraphBaseLayer):
    """DimNetPP Output Block as defined by `DimNetPP <https://arxiv.org/abs/2011.14115>`_ .

    Args:
        emb_size (list): List of node embedding dimension.
        out_emb_size (list): List of edge embedding dimension.
        num_dense (list): Number of dense layer for MLP.
        num_targets (int): Number of output target dimension. Defaults to 12.
        use_bias (bool, optional): Use bias. Defaults to True.
        kernel_initializer: Initializer for kernels. Default is 'glorot_orthogonal' with fallback 'orthogonal'.
        output_kernel_initializer: Initializer for last kernel. Default is 'zeros'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        activation (str): Activation function. Default is 'kgcnn>swish'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        pooling_method (str): Pooling method information for layer. Default is 'mean'.
    """

    def __init__(self, emb_size,
                 out_emb_size,
                 num_dense,
                 num_targets=12,
                 use_bias=True,
                 output_kernel_initializer="zeros", kernel_initializer='kgcnn>glorot_orthogonal',
                 bias_initializer='zeros',
                 activation='kgcnn>swish',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 pooling_method="sum",
                 **kwargs):
        """Initialize layer."""
        super(DimNetOutputBlock, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.emb_size = emb_size
        self.out_emb_size = out_emb_size
        self.num_dense = num_dense
        self.num_targets = num_targets
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_initializer": bias_initializer,
                       "bias_regularizer": bias_regularizer, "bias_constraint": bias_constraint, }

        self.dense_rbf = DenseEmbedding(emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.up_projection = DenseEmbedding(out_emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.dense_mlp = GraphMLP([out_emb_size] * num_dense, activation=activation,
                                  kernel_initializer=kernel_initializer, use_bias=use_bias, **kernel_args)
        self.dimnet_mult = LazyMultiply()
        self.pool = PoolingLocalEdges(pooling_method=self.pooling_method)
        self.dense_final = DenseEmbedding(num_targets, use_bias=False, kernel_initializer=output_kernel_initializer,
                                          **kernel_args)
        self.dropout = ks.layers.Dropout(rate=0.1)

    def build(self, input_shape):
        """Build layer."""
        super(DimNetOutputBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, tensor_index, state]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - rbf (tf.RaggedTensor): Edge distance basis of shape (batch, [M], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Updated node embeddings of shape (batch, [N], F_T).
        """
        # Calculate edge Update
        n_atoms, x, rbf, idnb_i = inputs
        g = self.dense_rbf(rbf, **kwargs)
        x = self.dimnet_mult([g, x], **kwargs)
        x = self.pool([n_atoms, x, idnb_i], **kwargs)
        x = self.up_projection(x, **kwargs)
        x = self.dense_mlp(x, **kwargs)
        x = self.dropout(x, training=kwargs.get('training', False))
        x = self.dense_final(x, **kwargs)
        return x

    def get_config(self):
        config = super(DimNetOutputBlock, self).get_config()
        conf_mlp = self.dense_mlp.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_mlp[x][0]})
        conf_dense_output = self.dense_final.get_config()
        config.update({"output_kernel_initializer": conf_dense_output["kernel_initializer"]})
        config.update({"pooling_method": self.pooling_method, "use_bias": self.use_bias})
        config.update({"emb_size": self.emb_size, "out_emb_size": self.out_emb_size, "num_dense": self.num_dense,
                       "num_targets": self.num_targets})
        return config

@ks.utils.register_keras_serializable(package='DimeNetPP', name='OutputProcessingLayer')
class OutputProcessingLayer(GraphBaseLayer):
    def __init__(self, num_splits=4, **kwargs):
        super(OutputProcessingLayer, self).__init__(**kwargs)
        self.num_splits = num_splits  # Number of output components (mu, logv, logalpha, logbeta)

    def call(self, inputs, **kwargs):
        out = inputs

        if len(out.shape) < 2:
            out = tf.expand_dims(out, axis=0)

        batch_size = tf.shape(out)[0]

        out = tf.reshape(out, [batch_size, -1, self.num_splits])

        return out

    def compute_output_shape(self, input_shape):
        # 输入形状是 (batch_size, features)
        if len(input_shape) < 2:
            input_shape = (None, input_shape[0])
        batch_size, features = input_shape
        output_shape = (batch_size, -1, self.num_splits)
        return output_shape

    def get_config(self):
        config = super(OutputProcessingLayer, self).get_config()
        config.update({"num_splits": self.num_splits})
        return config

model_default = {
    "name": "DimeNetPP",
    "inputs": [{"shape": [None], "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128,
                                 "embeddings_initializer": {"class_name": "RandomUniform",
                                                            "config": {"minval": -1.7320508075688772,
                                                                       "maxval": 1.7320508075688772}}}},
    "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
    "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
    "cutoff": 5.0, "envelope_exponent": 5,
    "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
    "num_targets": 64, "extensive": True, "output_init": "zeros",
    "activation": "swish", "verbose": 10,
    "output_embedding": "graph",
    "use_output_mlp": True,
    "output_mlp": {"use_bias": [True, False],
                   "units": [64, 12], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               emb_size: int = None,
               out_emb_size: int = None,
               int_emb_size: int = None,
               basis_emb_size: int = None,
               num_blocks: int = None,
               num_spherical: int = None,
               num_radial: int = None,
               cutoff: float = None,
               envelope_exponent: int = None,
               num_before_skip: int = None,
               num_after_skip: int = None,
               num_dense_output: int = None,
               num_targets: int = None,
               activation: str = None,
               extensive: bool = None,
               output_init: str = None,
               verbose: int = None,
               name: str = None,
               output_embedding: str = None,
               use_output_mlp: bool = None,
               output_mlp: dict = None
               ):
    """Make `DimeNetPP <https://arxiv.org/abs/2011.14115>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DimeNetPP.model_default`.

    .. note::
        DimeNetPP does require a large amount of memory for this implementation, which increase quickly with
        the number of connections in a batch.

    Inputs:
        list: `[node_attributes, node_coordinates, bond_indices, angle_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - node_coordinates (tf.RaggedTensor): Atomic coordinates of shape `(batch, None, 3)`.
            - bond_indices (tf.RaggedTensor): Index list for edges or bonds of shape `(batch, None, 2)`.
            - angle_indices (tf.RaggedTensor): Index list of angles referring to bonds of shape `(batch, None, 2)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.


    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        emb_size (int): Overall embedding size used for the messages.
        out_emb_size (int): Embedding size for output of :obj:`DimNetOutputBlock`.
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used inside the basis transformation.
        num_blocks (int): Number of graph embedding blocks or depth of the network.
        num_spherical (int): Number of spherical components in :obj:`SphericalBasisLayer`.
        num_radial (int): Number of radial components in basis layer.
        cutoff (float): Distance cutoff for basis layer.
        envelope_exponent (int): Exponent in envelope function for basis layer.
        num_before_skip (int): Number of residual layers in interaction block before skip connection
        num_after_skip (int): Number of residual layers in interaction block after skip connection
        num_dense_output (int): Number of dense units in output :obj:`DimNetOutputBlock`.
        num_targets (int): Number of targets or output embedding dimension of the model.
        activation (str, dict): Activation to use.
        extensive (bool): Graph output for extensive target to apply sum for pooling or mean otherwise.
        output_init (str, dict): Output initializer for kernel.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final :obj:`MLP`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation. Note that DimeNetPP originally defines the output dimension
            via `num_targets`. But this can be set to `out_emb_size` and the `output_mlp` be used for more
            specific control.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    bond_index_input = ks.layers.Input(**inputs[2])
    angle_index_input = ks.layers.Input(**inputs[3])

    # Atom embedding
    # n = generate_node_embedding(node_input, input_node_shape, input_embedding["nodes"])
    if len(inputs[0]["shape"]) == 1:
        n = EmbeddingDimeBlock(**input_embedding["node"])(node_input)
    else:
        n = node_input

    x = xyz_input
    edi = bond_index_input
    adi = angle_index_input

    # Calculate distances
    pos1, pos2 = NodePosition()([x, edi])
    d = NodeDistanceEuclidean()([pos1, pos2])
    rbf = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)(d)

    # Calculate angles
    v12 = LazySubtract()([pos1, pos2])
    a = EdgeAngle()([v12, adi])
    sbf = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                              envelope_exponent=envelope_exponent)([d, a, adi])

    # Embedding block
    rbf_emb = DenseEmbedding(emb_size, use_bias=True, activation=activation,
                             kernel_initializer="kgcnn>glorot_orthogonal")(rbf)
    n_pairs = GatherNodes()([n, edi])
    x = LazyConcatenate(axis=-1)([n_pairs, rbf_emb])
    x = DenseEmbedding(emb_size, use_bias=True, activation=activation, kernel_initializer="kgcnn>glorot_orthogonal")(x)
    ps = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                           output_kernel_initializer=output_init)([n, x, rbf, edi])

    # Interaction blocks
    add_xp = LazyAdd()
    for i in range(num_blocks):
        x = DimNetInteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip)(
            [x, rbf, sbf, adi])
        p_update = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                                     output_kernel_initializer=output_init)([n, x, rbf, edi])
        ps = add_xp([ps, p_update])

    if extensive:
        out = PoolingNodes(pooling_method="sum")(ps)
    else:
        out = PoolingNodes(pooling_method="mean")(ps)

    if use_output_mlp:
        out = MLP(**output_mlp)(out)

    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `DimeNetPP`.")

    model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input, angle_index_input],
                            outputs=out)

    return model


model_crystal_default = {
    "name": "DimeNetPP",
    "inputs": [{"shape": [None], "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
               {'shape': (None, 3), 'name': "edge_image", 'dtype': 'int64', 'ragged': True},
               {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
               ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128,
                                 "embeddings_initializer": {"class_name": "RandomUniform",
                                                            "config": {"minval": -1.7320508075688772,
                                                                       "maxval": 1.7320508075688772}}}},
    "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
    "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
    "cutoff": 5.0, "envelope_exponent": 5,
    "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
    "num_targets": 64, "extensive": True, "output_init": "zeros",
    "activation": "swish", "verbose": 10,
    "output_embedding": "graph",
    "use_output_mlp": True,
    "output_mlp": {"use_bias": [True, False],
                   "units": [64, 12], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_crystal_default)
def make_crystal_model(inputs: list = None,
                       input_embedding: dict = None,
                       emb_size: int = None,
                       out_emb_size: int = None,
                       int_emb_size: int = None,
                       basis_emb_size: int = None,
                       num_blocks: int = None,
                       num_spherical: int = None,
                       num_radial: int = None,
                       cutoff: float = None,
                       envelope_exponent: int = None,
                       num_before_skip: int = None,
                       num_after_skip: int = None,
                       num_dense_output: int = None,
                       num_targets: int = None,
                       activation: str = None,
                       extensive: bool = None,
                       output_init: str = None,
                       verbose: int = None,
                       name: str = None,
                       output_embedding: str = None,
                       use_output_mlp: bool = None,
                       output_mlp: dict = None
                       ):
    """Make `DimeNetPP <https://arxiv.org/abs/2011.14115>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DimeNetPP.model_crystal_default`.

    .. note::
        DimeNetPP does require a large amount of memory for this implementation, which increase quickly with
        the number of connections in a batch.

    Inputs:
        list: `[node_attributes, node_coordinates, bond_indices, angle_indices, edge_image, lattice]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - node_coordinates (tf.RaggedTensor): Atomic coordinates of shape `(batch, None, 3)`.
            - bond_indices (tf.RaggedTensor): Index list for edges or bonds of shape `(batch, None, 2)`.
            - angle_indices (tf.RaggedTensor): Index list of angles referring to bonds of shape `(batch, None, 2)`.
            - lattice (tf.Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)`.
            - edge_image (tf.RaggedTensor): Indices of the periodic image the sending node is located. The indices
                of and edge are :math:`(i, j)` with :math:`j` being the sending node.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.


    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        emb_size (int): Overall embedding size used for the messages.
        out_emb_size (int): Embedding size for output of :obj:`DimNetOutputBlock`.
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used inside the basis transformation.
        num_blocks (int): Number of graph embedding blocks or depth of the network.
        num_spherical (int): Number of spherical components in :obj:`SphericalBasisLayer`.
        num_radial (int): Number of radial components in basis layer.
        cutoff (float): Distance cutoff for basis layer.
        envelope_exponent (int): Exponent in envelope function for basis layer.
        num_before_skip (int): Number of residual layers in interaction block before skip connection
        num_after_skip (int): Number of residual layers in interaction block after skip connection
        num_dense_output (int): Number of dense units in output :obj:`DimNetOutputBlock`.
        num_targets (int): Number of targets or output embedding dimension of the model.
        activation (str, dict): Activation to use.
        extensive (bool): Graph output for extensive target to apply sum for pooling or mean otherwise.
        output_init (str, dict): Output initializer for kernel.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final :obj:`MLP`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation. Note that DimeNetPP originally defines the output dimension
            via `num_targets`. But this can be set to `out_emb_size` and the `output_mlp` be used for more
            specific control.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    bond_index_input = ks.layers.Input(**inputs[2])
    angle_index_input = ks.layers.Input(**inputs[3])
    edge_image = ks.layers.Input(**inputs[4])
    lattice = ks.layers.Input(**inputs[5])

    # Atom embedding
    # n = generate_node_embedding(node_input, input_node_shape, input_embedding["nodes"])
    if len(inputs[0]["shape"]) == 1:
        n = EmbeddingDimeBlock(**input_embedding["node"])(node_input)
    else:
        n = node_input

    x = xyz_input
    edi = bond_index_input
    adi = angle_index_input

    # Calculate distances
    pos1, pos2 = NodePosition()([x, edi])
    pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice])
    d = NodeDistanceEuclidean()([pos1, pos2])
    rbf = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)(d)

    # Calculate angles
    v12 = LazySubtract()([pos1, pos2])
    a = EdgeAngle()([v12, adi])
    sbf = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                              envelope_exponent=envelope_exponent)([d, a, adi])

    # Embedding block
    rbf_emb = DenseEmbedding(emb_size, use_bias=True, activation=activation,
                             kernel_initializer="kgcnn>glorot_orthogonal")(rbf)
    n_pairs = GatherNodes()([n, edi])
    x = LazyConcatenate(axis=-1)([n_pairs, rbf_emb])
    x = DenseEmbedding(emb_size, use_bias=True, activation=activation, kernel_initializer="kgcnn>glorot_orthogonal")(x)
    ps = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=4,
                           output_kernel_initializer=output_init)([n, x, rbf, edi])

    # Interaction blocks
    add_xp = LazyAdd()
    for i in range(num_blocks):
        x = DimNetInteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip)(
            [x, rbf, sbf, adi])
        p_update = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=4,
                                     output_kernel_initializer=output_init)([n, x, rbf, edi])
        ps = add_xp([ps, p_update])

    if extensive:
        out = PoolingNodes(pooling_method="sum")(ps)
    else:
        out = PoolingNodes(pooling_method="mean")(ps)

    if use_output_mlp:
        out = MLP(**output_mlp)(out)

    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `DimeNetPP`.")

    out = OutputProcessingLayer(num_splits=4)(out)

    model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input, angle_index_input, edge_image, lattice],
                            outputs=out)

    return model
