import torch
import argparse
from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph.defense import GAT
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PrePtbDataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='perturbation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# use data splist provided by prognn
data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

gat = GAT(nfeat=features.shape[1],
      nhid=8, heads=8,
      nclass=labels.max().item() + 1,
      dropout=0.5, device=device)
gat = gat.to(device)


# test on clean graph
# print('==================')
# print('=== train on clean graph ===')


# gat.fit(pyg_data, verbose=True) # train with earlystopping
# gat.test()

# load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
print('==================')
print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')
if args.ptb_rate==0:
        perturbed_adj=adj
else:
        perturbed_data = PrePtbDataset(root='/tmp/',
                name=args.dataset,
                attack_method='meta',
                ptb_rate=args.ptb_rate)
        perturbed_adj = perturbed_data.adj
data.adj=perturbed_adj
pyg_data = Dpr2Pyg(data)
# pyg_data.update_edge_index(perturbed_adj) # inplace operation
gat.fit(pyg_data, verbose=True) # train with earlystopping
gat.test()



