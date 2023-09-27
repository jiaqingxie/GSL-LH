from msilib.schema import Error
import time
import argparse
import numpy as np
import torch
import torch_sparse
from model.gcn import GCN
from model.gat import GAT
from model.gin import GIN

# from deeprobust.graph.defense import GCN, ProGNN
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import preprocess, encode_onehot, get_train_val_test
from deeprobust.graph.data import Dpr2Pyg, Pyg2Dpr
import ipdb
from GLYSL import GLYSL
from GLYSL_GATGIN import GLYSL_GATGIN
from deeprobust.graph.defense import GAT as GAT_t
from model.gin_self import GIN_self

# Training settings
parser = argparse.ArgumentParser()
# Model Config
parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
parser.add_argument(
    "--only_gcn",
    action="store_true",
    default=False,
    help="test the performance of gcn without other components",
)


parser.add_argument(
    "--gcn",
    action="store_true",
    default=False,
    help="test the performance of gat without other components",
)


# Hyper Parameter
parser.add_argument("--ugs", action="store_true", default=True, help="find the lottery ticket in graph and use the subnetwork to pursue the better performance")
parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")  # default:42
parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate.")
parser.add_argument("--lr_mask", type=float, default=1, help="lr for training mask")
parser.add_argument("--lr_adj", type=float, default=1e-2, help="lr for training mask")

parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=16, help="Number of hidden units.")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability).")
parser.add_argument(
    "--dataset",
    type=str,
    default="cora",
    choices=["cora", "cora_ml", "citeseer", "polblogs", "pubmed","ogbn-arxiv"],
    help="dataset",
)
parser.add_argument("--attack", type=str, default="meta", choices=["no", "meta", "random", "nettack"])
parser.add_argument("--ptb_rate", type=float, default=0, help="noise ptb_rate")
# Weight
parser.add_argument("--alpha", type=float, default=1e-5, help="weight of l1 norm")
parser.add_argument("--beta", type=float, default=1.5, help="weight of nuclear norm")
parser.add_argument("--epsilon", type=float, default=1e-4, help="weight of adjacency L1 loss")

parser.add_argument("--gamma", type=float, default=1, help="weight of gcn")
parser.add_argument("--lambda_", type=float, default=0, help="weight of feature smoothing")
parser.add_argument("--phi", type=float, default=5e-7, help="weight of symmetric loss")
parser.add_argument("--omicron", type=float, default=5e-7, help="weight of gc param L1 loss")
parser.add_argument("--delta", type=float, default=1e-2, help="weight of pro loss")
# Inner & Outer
parser.add_argument("--inner_steps", type=int, default=2, help="steps for inner optimization")
parser.add_argument("--outer_steps", type=int, default=1, help="steps for outer optimization")

parser.add_argument("--symmetric", action="store_true", default=False, help="whether use symmetric matrix")
parser.add_argument("--korder", type=int, default=2, help="find the k-order of the node")
parser.add_argument("--epoch", type=int, default=1, help="train the mask and param simultaneously")
parser.add_argument("--epoch1", type=int, default=500, help="train the param")
parser.add_argument("--epoch2", type=int, default=500, help="train the mask")
parser.add_argument("--epoch3", type=int, default=500, help="train with mask")
parser.add_argument("--train_adj_only", action="store_true")
parser.add_argument("--train_weight_only", action="store_true")
parser.add_argument("--adj_sparsity", type=float, default=-1, help="adj_sparsity")
parser.add_argument("--weight_sparsity", type=float, default=-1, help="weight_sparsity")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

# print(args)

# Here the random seed is to split the train/val/test data,
# we need to set the random seed to be the same as that when you generate the perturbed graph
# but now change the setting from nettack to prognn which directly loads the prognn splits
# data = Dataset(root="../data/", name=args.dataset, setting='nettack', seed=15)

if args.dataset in ['cora', 'citeseer', 'cora_ml', 'polblogs',
                'pubmed', 'acm', 'blogcatalog', 'uai', 'flickr']:
    data = Dataset(root="../data/", name=args.dataset, setting="prognn")
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
else:
    pass

if args.attack == "no":
    perturbed_adj = adj

if args.attack == "random":
    from deeprobust.graph.global_attack import Random

    # to fix the seed of generated random attack, you need to fix both np.random and random
    # you can uncomment the following code
    # import random; random.seed(args.seed)
    # np.random.seed(args.seed)
    attacker = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    attacker.attack(adj, n_perturbations, type="add")
    perturbed_adj = attacker.modified_adj

if args.attack == "meta" or args.attack == "nettack":
    perturbed_data = PrePtbDataset(
        root="../data/",
        name=args.dataset,
        attack_method=args.attack,
        ptb_rate=args.ptb_rate,
    )
    perturbed_adj = perturbed_data.adj
    if args.attack == "nettack":
        idx_test = perturbed_data.target_nodes

np.random.seed(args.seed)
torch.manual_seed(args.seed)
model = GCN(
    nfeat=features.shape[1],
    nhid=args.hidden,
    nclass=labels.max().item() + 1,
    lr=args.lr,
    dropout=args.dropout,
    device=device,
)
# GCN
if args.only_gcn:
    perturbed_adj, features, labels = preprocess(
        perturbed_adj,
        features,
        labels,
        preprocess_adj=False,
        sparse=True,
        device=device,
    )
    model.fit(
        features,
        perturbed_adj,
        labels,
        idx_train,
        idx_val,
        verbose=True,
        train_iters=1000,
    )

    model.test(idx_test)
elif args.gcn:
    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)
    gogo = GLYSL(model, args, device)
    gogo.fit(features, perturbed_adj, labels, idx_train, idx_val,idx_test)
    gogo.test(features, labels, idx_test)
else:
    print("Error, please set true model")