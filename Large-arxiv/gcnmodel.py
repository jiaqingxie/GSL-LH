import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from model.maskedlinear import MaskedLinear
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.optim as optim
from copy import deepcopy
import torch.nn.functional as F
from deeprobust.graph import utils
import torch.nn as nn
import math
from torch_geometric.nn.dense.linear import Linear
class GCNConvMasked(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True,with_bias=True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        in_features=in_channels
        out_features=out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_mask = Parameter(torch.FloatTensor(in_features, out_features))
        

        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias_mask = Parameter(torch.FloatTensor(out_features))
            
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # Set Mask to 1
        nn.init.constant_(self.weight_mask.data, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            nn.init.constant_(self.bias_mask.data, 1)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        self._cached_edge_index = None
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        masked_weight = torch.clamp(self.weight_mask,0,1) * self.weight
        x = torch.mm(x, masked_weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            masked_bias = torch.clamp(self.bias_mask,0,1 )* self.bias
            out += masked_bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if  edge_weight is None else  edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class GCN_Masked(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN_Masked, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConvMasked(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvMasked(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConvMasked(hidden_channels, out_channels, cached=True))

        self.dropout = dropout
        self.r_, self.l_, self.b_ = -0.1, 1.1, 2 / 3
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    def replace_layers_with_masked(self, out_w_per_mask=1, in_w_per_mask=1, mask_p=0.9, verbose=False):
    # """
    # Replaces layers with their masked versions.
    # out_w_per_mask: the number of output dims covered by a single mask parameter
    # in_w_per_mask: the same as above but for input dims

    # ex: (1,1) for weight masking
    #     (768,1) for neuron masking
    #     (768, 768) for layer masking
    # """

        def replace_layers(layer_names, parent_types, replacement):
            for module_name, module in self.named_modules():
                for layer_name in layer_names:
                    if hasattr(module, layer_name) and type(module) in parent_types:
                        layer = getattr(module, layer_name)
                        setattr(module, layer_name, replacement(layer))
                        if verbose:
                            print("Replaced {} in {}".format(layer_name, module_name))

        # replace_layers(('query', 'key', 'value', 'dense'),
        #                (BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput),
        #                lambda x: MaskedLinear.from_layer(x, out_w_per_mask, in_w_per_mask, mask_p))

        replace_layers(["lin"],
                       (GCNConvMasked,),
                       lambda x: MaskedLinear.from_layer(x, out_w_per_mask, in_w_per_mask, mask_p))
    def compute_binary_pct(self):

        total, n = 0, 0
        for k, v in self.named_parameters():
            if 'mask_scores' in k:
                v = v.detach().cpu().numpy().flatten()
                v = 1 / (1 + np.exp(-v))  # sigmoid
                # total += np.sum(v < 0.01) + np.sum(v > 0.99)
                total += np.sum(v < (-self.r_)/(self.l_-self.r_)) + np.sum(v > (1-self.r_)/(self.l_-self.r_))
                n += v.size
        return total / n

    def compute_half_pct(self):
        total, n = 0, 0
        for k, v in self.named_parameters():
            if 'mask' in k:
                v = v.detach().cpu().numpy().flatten()
                total += np.sum(v<0.5)
                n += v.size
        return total / n
    def compute_zero_pct(self):

        total, n = 0, 0
        for k, v in self.named_parameters():
            if 'mask' in k:
                v = v.detach().cpu().numpy().flatten()
                # total += np.sum(v < 0.01) + np.sum(v > 0.99)
                total += np.sum(v < 0.01)
                n += v.size
        return total / n
    def compute_one_pct(self):

        total, n = 0, 0
        for k, v in self.named_parameters():
            if 'mask' in k:
                v = v.detach().cpu().numpy().flatten()
                # total += np.sum(v < 0.01) + np.sum(v > 0.99)
                total += np.sum(v > 0.99)
                n += v.size
        return total / n