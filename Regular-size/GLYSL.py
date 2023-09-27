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
import torch.nn.utils.prune as prune
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
    device=adj.device
    adj=adj.to("cpu")
    sum, mul = adj, adj
    for _ in range(K - 1):
        mul = torch.matmul(mul, adj)
        sum = sum + mul
    return (sum > 0).long().to(device)


def find_K_orderfathergraph_sparse(adj, K):

    sum, mul, adj_ = SparseTensor.from_dense(adj), SparseTensor.from_dense(adj), SparseTensor.from_dense(adj)
    for _ in range(K - 1):
        mul = torch_sparse.matmul(mul, adj_)
        sum = torch_sparse.add(sum, mul)
    row, col, val = sum.coo()
    # edge index
    # return torch.cat((torch.unsqueeze(row, 0), torch.unsqueeze(col, 0)), dim=0)
    return SparseTensor(row=row, col=col, value=(val > 0).long())


class GLYSL:
    """GLYSL (Graph Lottery Structure learning).

    Parameters
    ----------
    model:
        model: The backbone GNN model in GLYSL
    args:
        model configs
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
    See details in https://github.com/ChandlerBang/Pro-GNN.

    """

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)

    def fit(self, features, adj, labels, idx_train, idx_val,idx_test, **kwargs):
        """Train Pro-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        args = self.args
        # Optimizer
        self.optimizer = optim.Adam([p for n, p in self.model.named_parameters() if "mask" not in n and p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_state = self.optimizer.state_dict().copy()
        self.mask_optimizer = optim.Adam([p for n, p in self.model.named_parameters() if "mask" in n and p.requires_grad], lr=args.lr_mask, weight_decay=args.weight_decay)

        estimator = EstimateAdj(find_K_orderfathergraph(adj, K=args.korder), symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.adj_optimizer = optim.SGD(estimator.parameters(), momentum=0.9, lr=args.lr_adj)

        # warnings.warn("If you find the nuclear proximal operator runs too slow, you can modify line 77 to use prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        # self.optimizer_nuclear = PGD(
        #     estimator.parameters(),
        #     proxs=[prox_operators.prox_nuclear],
        #     lr=args.lr_adj,
        #     alphas=[args.beta],
        # )
        self.optimizer_nuclear = None

        # Train model
        t_total = time.time()

        orginal_weight = copy.deepcopy(self.model.state_dict())
        rewind = copy.deepcopy({k: v for k, v in self.model.state_dict().items() if "mask" not in k})
        ###train_model

            # Rotational training
            # if _ % step == 0:
            #     cur_model_state = self.model.state_dict()
            #     cur_model_state.update(rewind)
            #     self.model.load_state_dict(cur_model_state)

        for epoch in range(int(args.epoch1)):
            self.train_gcn(epoch, features, estimator.estimated_adj, labels, idx_train, idx_val,idx_test)
            # device2=torch.device("cuda:1")
            # features.to(device2)
            # self.estimator.to(device2)
            # labels.to(device2)
            # self.model.to(device2)
            # self.estimator.device=device2
        torch.cuda.empty_cache()
        # self.best_val_acc = 0
        # self.best_val_loss = 10
        for epoch in range(int(args.epoch2)):
            self.train_mask(epoch, features, adj, labels, idx_train, idx_val,idx_test)

        # Load original state
        cur_model_state = self.weights
        cur_model_state.update(rewind)  # reset parameter
        self.model.load_state_dict(cur_model_state)  # load state_dict

        estimator.estimated_adj = nn.Parameter(self.best_graph)
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.optimizer.load_state_dict(self.optimizer_state)
         ###########retrain lottery###########################################
        from draw_retrain_ticket import draw_ticket_mask,init_mask_score
        if not args.train_adj_only:
            if args.weight_sparsity<0:
                pass
            else:
                weight_sparsity=args.weight_sparsity


                ticket_mask = draw_ticket_mask(self.model, weight_sparsity)

                init_mask_score(self.model, ticket_mask)
        if not args.train_weight_only:
        #prune edge_index##################
            if args.adj_sparsity<0:
                pass
            else:
                estimator=self.estimator
                
                adj_sparsity=args.adj_sparsity
                proto_adj=self.estimator.estimated_adj>0
                
                proto_nnz=torch.sum(proto_adj)

                print("before prune: nnz=", proto_nnz)

                pruned_nnz=int(proto_nnz*(1-adj_sparsity))
                true_prune_sparsity=1-pruned_nnz/proto_adj.size(0)/proto_adj.size(0)
                # estimator.mask
                ################positive mask_scores
                module =  self.estimator.estimated_adj
                module.data = torch.sigmoid(module.data)
                ######Prune mask_scores##############
                parameters_to_prune = []

                parameters_to_prune.append((estimator, 'estimated_adj'))



                parameters_to_prune = tuple(parameters_to_prune)

                prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=true_prune_sparsity,
                        )
                        ################extract mask scores_mask
                mask_scores_mask_dict = {}
                model_dict = estimator.state_dict()
                for key in model_dict.keys():
                    if 'estimated_adj_mask' in key:
                        mask_scores_mask_dict[key] = model_dict[key]
                ###########Init mask _scores##########
                module = estimator.estimated_adj
                module_mask = 'estimated_adj_mask'
                mask = mask_scores_mask_dict[module_mask]
                module.data = mask
                print("after prune: nnz=",torch.sum(estimator.estimated_adj))


    ####################################
        for _ in range(args.epoch3):
            torch.cuda.empty_cache()
            self.train_gcn(_, features, self.best_graph, labels, idx_train, idx_val,idx_test)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

    # think about how to train_mask
    # def train_mask(self, epoch, features, adj, labels, idx_train, idx_val):

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val,idx_test):
        args = self.args
        estimator = self.estimator
        adj = estimator.normalize()
        # print((adj > 0).sum())
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward(retain_graph=True)
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        with torch.no_grad():
            output = self.model(features, adj)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test=accuracy(output[idx_test], labels[idx_test])
            if acc_val > self.best_val_acc and acc_val != nan:
                self.best_val_acc = acc_val
                self.best_graph = adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if args.debug:
                    print("=== saving current graph/gcn, best_val_acc: %s ===" % self.best_val_acc.item())

            print(
                colored("FatherGraph:", "red"),
                "Epoch: {:04d}".format(epoch + 1),
                "acc_train: {:.4f}".format(acc_train.item()),
                "loss_val: {:.4f}".format(loss_val.item()),
                "acc_val: {:.4f}".format(acc_val.item()),
                "time: {:.4f}s".format(time.time() - t),
                "acc_test: {:.4f}".format(acc_test.item()),
            )
            """
            if loss_val < self.best_val_loss:
                self.best_val_loss = loss_val
                self.best_graph = adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if args.debug:
                    print(
                        f"\t=== saving current graph/gcn, best_val_loss: %s"
                        % self.best_val_loss.item()
                    )
            """
            if args.debug:
                if epoch % 1 == 0:
                    print(
                        "Epoch: {:04d}".format(epoch + 1),
                        "loss_train: {:.4f}".format(loss_train.item()),
                        "acc_train: {:.4f}".format(acc_train.item()),
                        "loss_val: {:.4f}".format(loss_val.item()),
                        "acc_val: {:.4f}".format(acc_val.item()),
                        "time: {:.4f}s".format(time.time() - t),
                    )

    def train_mask(self, epoch, features, adj, labels, idx_train, idx_val,idx_test):
        # train
        estimator = self.estimator
        father_graph = estimator.estimated_adj
        args = self.args

        if args.debug:
            print("\n=== train_mask ===")
        t = time.time()

        estimator.train()
        self.model.train()

        self.adj_optimizer.zero_grad()
        self.mask_optimizer.zero_grad()

        # |m_g|
        loss_mask_adj = torch.norm(estimator.estimated_adj, 1)
        # |m_theta|
        loss_mask_param = torch.norm(self.model.gc1.weight_mask, 1) + torch.norm(self.model.gc1.bias_mask, 1) + torch.norm(self.model.gc2.weight_mask, 1) + torch.norm(self.model.gc2.bias_mask, 1)

        normalized_adj = estimator.normalize()

        output = self.model(features, normalized_adj)
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        loss_symmetric = torch.norm(estimator.estimated_adj - estimator.estimated_adj.t(), p="fro")

        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0

        loss = args.gamma * loss_gcn + args.phi * loss_symmetric + args.epsilon * loss_mask_adj + args.lambda_ * loss_smooth_feat
        # + args.omicron * loss_mask_adj + args.delta * loss_mask_param
        loss.backward()
        self.model.gc1.weight_mask.grad.add_(args.omicron * torch.sign(self.model.gc1.weight_mask))
        self.model.gc1.bias_mask.grad.add_(args.omicron * torch.sign(self.model.gc1.bias_mask))
        self.model.gc2.weight_mask.grad.add_(args.delta * torch.sign(self.model.gc2.weight_mask))
        self.model.gc2.bias_mask.grad.add_(args.delta * torch.sign(self.model.gc2.bias_mask))

        self.adj_optimizer.step()
        self.mask_optimizer.step()

        # scale to [0, 1] and restrict to the original graph
        estimator.estimated_adj.data.copy_(torch.clamp(estimator.estimated_adj, min=0, max=1) * father_graph)

        # evaluate
        self.model.eval()
        normalized_adj = estimator.normalize()
        # if args.debug:
        # print((normalized_adj > 0).sum())
        # TO DO evaluation
        with torch.no_grad():
            output = self.model(features, normalized_adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test=accuracy(output[idx_test], labels[idx_test])
            print(
                colored("SubGraph", "blue"),
                "Epoch: {:04d}".format(epoch + 1),
                "acc_train: {:.4f}".format(acc_train.item()),
                "loss_val: {:.4f}".format(loss_val.item()),
                "acc_val: {:.4f}".format(acc_val.item()),
                "time: {:.4f}s".format(time.time() - t),
                "acc_test: {:.4f}".format(acc_test.item()),
            )
            # print(normalized_adj.detach())
            # ipdb.set_trace()

            if acc_val > self.best_val_acc and acc_val != nan:
                self.best_val_acc = acc_val
                self.best_graph = normalized_adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if args.debug:
                    print("=== saving current graph/gcn, best_val_acc: %s ===" % self.best_val_acc.item())

            """
            if loss_val < self.best_val_loss:
                self.best_val_loss = loss_val
                self.best_graph = normalized_adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if args.debug:
                    print(
                        f"\t=== saving current graph/gcn, best_val_loss: %s"
                        % self.best_val_loss.item()
                    )
            """

            if args.debug:
                if epoch % 1 == 0:
                    print(
                        "Epoch: {:04d}".format(epoch + 1),
                        "loss_gcn: {:.4f}".format(loss_gcn.item()),
                        "loss_symmetric: {:.4f}".format(loss_symmetric.item()),
                        "delta_l1_norm: {:.4f}".format(torch.norm(estimator.estimated_adj - adj, 1).item()),
                        "loss_mask_adj: {:.4f}".format(loss_mask_adj.item()),
                        "loss_mask_param: {:.4f}".format(loss_mask_param.item()),
                    )

    # Add Mask

    def test(self, features, labels, idx_test):
        """Evaluate the performance of GLYSL on test set"""
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.estimator.normalize()
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print(
            "\tTest set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
        )
        return acc_test.item()

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj) / 2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device="cpu"):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):
        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t()) / 2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx


if __name__ == "__main__":
    input = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
    g = find_K_orderfathergraph(input, K=2)
    print(g)
