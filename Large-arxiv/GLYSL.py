from math import nan
import time
import numpy as np
from copy import deepcopy
import torch
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.utils import accuracy
from deeprobust.graph.defense import GCN, ProGNN
from deeprobust.graph.defense.pgd import PGD, prox_operators
from torch.utils.tensorboard import SummaryWriter
import warnings
import copy
from termcolor import colored
from torch_sparse import SparseTensor
import ipdb

"""
Procedures
1. train gcn
2. find mask
3. set original weight
4. train subgraph
"""

def filter_neighbor(adj):

    pass
def find_K_orderfathergraph(adj, K):

    sum, mul = adj, adj
    for _ in range(K - 1):
        mul = torch.matmul(mul, adj)
        sum = sum + mul
    import ipdb
    ipdb.set_trace()
    return (sum > 0).long()


def find_K_orderfathergraph_sparse(adj:SparseTensor, K):

    sum, mul, adj_ = adj,adj,adj
    for _ in range(K - 1):
        mul = torch_sparse.matmul(mul, adj_)
        sum = torch_sparse.add(sum, mul)
    row, col, val = sum.coo()

    nonzeroindex=torch.nonzero(val).squeeze()

    row=torch.index_select(row,-1,nonzeroindex)
    col=torch.index_select(col,-1,nonzeroindex)
    # edge index
    # return torch.cat((torch.unsqueeze(row, 0), torch.unsqueeze(col, 0)), dim=0)
    return SparseTensor(row=row, col=col, value=torch.ones(row.size()))



class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, edge_index):
        super(EstimateAdj, self).__init__()
        self.edge_index=edge_index
        self.mask=nn.Parameter(torch.FloatTensor(edge_index.size(1), ))
        self.device=edge_index.device
        self.reset_parameters()
    # def _init_estimation(self, adj):
    #     with torch.no_grad():
    #         n = len(adj)
    #         self.estimated_adj.data.copy_(adj)
    def reset_parameters(self):
        # Set Mask to 1
        nn.init.constant_(self.mask.data, 1)
    def produce_mask_reshaped(self):
        mask = torch.clamp(self.mask,min=0,max=1)

        return mask
    def selectnonzeroweight(self):
        edge_weight_mask=self.produce_mask_reshaped()
        self.edge_index=self.edge_index.to(self.device)

        nonzeroindex=torch.nonzero(edge_weight_mask).squeeze().to(self.device)

        edge_index=torch.index_select(self.edge_index,-1,nonzeroindex)
        edge_weight=torch.index_select(edge_weight_mask,-1,nonzeroindex)
        return edge_index, edge_weight
    def forward(self):
        
        return self.selectnonzeroweight()




if __name__ == "__main__":
    # input = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
    # g = find_K_orderfathergraph(input, K=2)
    # print(g)
    edge_index=torch.Tensor([[0,1,2,3,4,5,6,7],[7,6,5,4,3,2,1,0]])

    estimate=EstimateAdj(edge_index)
    estimate.edge_weight_mask.data=torch.Tensor([1,-1,-2,2,1,1,0.5,1])
    i,e=estimate.forward()
