import torch
import torch_sparse
import numpy as np
from torch_sparse import SparseTensor
from deeprobust.graph.utils import normalize_adj, normalize_feature
from torch_sparse.matmul import matmul
from copy import deepcopy
from tqdm import tqdm
import multiprocessing
import timeit
import operator


def sparse_mx_to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row)
    sparsecol = torch.LongTensor(sparse_mx.col)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    return SparseTensor(row=sparserow, col=sparsecol, value=sparsedata)


def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=True, sparse=True, device="cpu"):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor, and normalize the input data.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    preprocess_adj : bool
        whether to normalize the adjacency matrix
    preprocess_feature : bool
        whether to normalize the feature matrix
    sparse : bool
       whether to return sparse tensor
    device : str
        'cpu' or 'cuda'
    """

    if preprocess_adj:
        adj = normalize_adj(adj)

    if preprocess_feature:
        features = normalize_feature(features)

    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse(adj)
        features = sparse_mx_to_torch_sparse(features)
    else:
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())
    return adj.to(device), features.to(device), labels.to(device)


def Add_edge_random(adj, features, rate=0.001):
    """
    Add random edges to the graph.
    """
    row_size, col_size = adj.size(dim=0), adj.size(dim=1)
    assert row_size == col_size, "Adj Dim is not equal"
    add_edge_num = int(row_size * col_size * rate)
    row, col = torch.randint(0, row_size, (add_edge_num,), dtype=torch.long), torch.randint(0, col_size, (add_edge_num,), dtype=torch.long)
    val = torch.ones((add_edge_num,), dtype=torch.long)
    add_adj = SparseTensor(row=row, col=col, value=val)
    row, col, val = torch_sparse.add(adj, add_adj).coo()

    # torch.save(SparseTensor(row=row, col=col, value=torch.ones_like(row, dtype=torch.long)),"/remote-home/share/zyyin/Graph/Arxiv_Graph/576_random_arxiv_5.pt")
    return SparseTensor(row=row, col=col, value=torch.ones_like(row, dtype=torch.long))
def Add_edge_feature(adj, features, rate=0.001,att_matrix=None,number=0,theta=9.5):
    """
    Add random edges to the graph.
    """
    print(adj)
    a=att_matrix > theta
    originalnnz=adj.coo()[0].size(0)
    edge_index=torch.nonzero(a).T
    add_adj=SparseTensor.from_edge_index(edge_index)
    nnz=edge_index.size(1)
    print(nnz)

    val = torch.ones((nnz,), dtype=torch.long)

    add_adj.set_value_(val)
    # add_adj = SparseTensor(row=row, col=col, value=val)
    row, col, val = torch_sparse.add(adj, add_adj).coo()
    
    outsparse=SparseTensor(row=row, col=col, value=torch.ones_like(row, dtype=torch.long))
    
    outnnz=row.size(0)
    print(outsparse)
    number=str(number)
    theta=str(theta)
    torch.save(outsparse,f"../data/makingadj/lotteryadj/featureadj_{number}_{theta}_{originalnnz}_{nnz}_{outnnz}.pt")
    return outsparse



def Add_edge_Lottery(adj, features, rate=0.001,att_matrix=None):
    """
    Add edges to the graph through Lottery algorithm.
    """
    row_size, col_size = adj.size(dim=0), adj.size(dim=1)
    assert row_size == col_size, "Adj Dim is not equal"
    num_Lottery = int(row_size * rate)
    if att_matrix is not None:
        Lottery = torch.cumsum(att_matrix.softmax(-1), dim=-1).squeeze()
    else:
        att_matrix=torch.mm(features, features.T)
        torch.save(att_matrix,f"../data/makingadj/arxiv_feature_att_matrix.pt")
        Lottery = torch.cumsum(att_matrix.softmax(-1), dim=-1).squeeze()
    row = torch.arange(row_size).repeat(num_Lottery, 1).T.reshape(-1).long()
    print("reading att_matrix complete")
    # winning_tickets = torch.rand((row_size, num_Lottery))
    # col = (winning_tickets[..., None] < Lottery[:, None]).eq(0).sum(dim=-1).reshape(-1).long()
    col = []
    for _ in tqdm(range(num_Lottery)):
        winning_tickets = torch.rand((row_size, 1))
        col.append((winning_tickets < Lottery).eq(0).sum(-1))
    col = torch.stack(col, dim=1).reshape(-1).long()
    val = torch.ones_like(row, dtype=torch.long)
    add_adj = SparseTensor(row=row, col=col, value=val)
    row, col, val = torch_sparse.add(adj, add_adj).coo()
    rate=str(rate)
    outsparse=SparseTensor(row=row, col=col, value=torch.ones_like(row, dtype=torch.long))
    torch.save(outsparse, f"../data/makingadj/lotteryadj/lotteryadj_5_{rate}.pt")
    print(outsparse)
    return outsparse
if __name__=="__main__":
    import torch_geometric.transforms as T
    from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
    
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=T.ToSparseTensor(), root="../data")
    data = dataset[0]
    adj=data.adj_t.to_symmetric()
    split_idx = dataset.get_idx_split()
    test_idx = split_idx["test"]
    print(test_idx)
    torch.save(test_idx,"test_idx.pt")
    # torch.save(adj,"original_adj.pt")
    # datapath="../data/ogbdata_attacked/arxiv_adj/arxiv_adj_1.pt"
    # # datapath="../data/ogbdata_attacked/arxiv_adj/arxiv_adj_2.pt"
    # # datapath="../data/ogbdata_attacked/arxiv_adj/arxiv_adj_3.pt"
    # # datapath="../data/ogbdata_attacked/arxiv_adj/arxiv_adj_4.pt"
    # # datapath="../data/ogbdata_attacked/arxiv_adj/arxiv_adj_5.pt"
    # adj=torch.load(datapath).to("cpu")
    # row,col,val=adj.coo()
    # adj.set_value_(torch.ones(row.size()))

    # attr_matrix=torch.load("arxiv_feature_att_matrix.pt").to("cpu")
    # Add_edge_feature(adj, data.x, rate=0.00001,att_matrix=attr_matrix,number=1,theta=10)
    