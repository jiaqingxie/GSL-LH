import argparse

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import GCNConv, SAGEConv
import copy

import torch.optim as optim
import torch_sparse
from logger import Logger
from gcnmodel import GCN_Masked
from GLYSL import EstimateAdj, find_K_orderfathergraph_sparse
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.data import Dpr2Pyg, Pyg2Dpr
import torch.nn.utils.prune as prune

import numpy as np
import random
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer, edge_weight=None):
    model.train()
    if edge_weight is not None:
        optimizer.zero_grad()
        out = model(data.x, data.adj_t,edge_weight)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        loss.backward(retain_graph=True)
        optimizer.step()

        return loss.item()

    else:

        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        loss.backward(retain_graph=True)
        optimizer.step()

        return loss.item()


def train_mask(model, data, train_idx, optimizer, estimator, optimizer_more, alpha, beta, gamma):

    model.train()
    estimator.train()
    if optimizer is not None:
        optimizer.zero_grad()

    if optimizer_more is not None:
        optimizer_more.zero_grad()
    edge_index,edge_weight = estimator.forward()

    out = model(data.x, edge_index, edge_weight)[train_idx]
    # out = model(data.x, data.adj_t)[train_idx]

    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss_mask_adj = torch.norm(edge_weight, 1)
    mask_weight = [p for n, p in model.named_parameters() if "mask" in n and p.requires_grad]  ###adj_norm
    loss_mask_param = sum([torch.norm(x, 1) for x in mask_weight])  ##weight_norm
    totalloss = alpha * loss+ beta * loss_mask_adj + gamma * loss_mask_param 
    totalloss.backward(retain_graph=True)
    if optimizer is not None:
        optimizer.step()
    if optimizer_more is not None:
        optimizer_more.step()


    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, estimator=None):
    model.eval()
    if estimator:
        estimator.eval()
        edge_index,edge_weight=estimator.forward()
        out = model(data.x, edge_index,edge_weight)
    else:
        out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({"y_true": data.y[split_idx["train"]], "y_pred": y_pred[split_idx["train"]],})["acc"]
    valid_acc = evaluator.eval({"y_true": data.y[split_idx["valid"]], "y_pred": y_pred[split_idx["valid"]],})["acc"]
    test_acc = evaluator.eval({"y_true": data.y[split_idx["test"]], "y_pred": y_pred[split_idx["test"]],})["acc"]

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description="OGBN-Arxiv (GNN)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--use_sage", action="store_true")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--korder", type=int, default=2, help="find the k-order of the node")
    parser.add_argument("--epoch", type=int, default=1, help="train the mask and param simultaneously")
    parser.add_argument("--epoch1", type=int, default=500, help="train the param")
    parser.add_argument("--epoch2", type=int, default=500, help="train the mask")
    parser.add_argument("--epoch3", type=int, default=500, help="train with mask")
    parser.add_argument("--alpha", type=float, default=1, help="weight of lossgcn")
    parser.add_argument("--beta", type=float, default=0, help="weight of l1 norm of adj")
    # parser.add_argument("--epsilon", type=float, default=1, help="weight of adjacency L1 loss")
    parser.add_argument("--weight_decay", type=int, default=0.01)
    parser.add_argument("--gamma", type=float, default=0, help="weight of l1 norm of weightmask")
    parser.add_argument("--lr_adj", type=float, default=0.1)
    parser.add_argument("--lr_weight", type=float, default=0.1)
    parser.add_argument("--gcn_masked", action="store_true")
    parser.add_argument("--attack", action="store_true")
    parser.add_argument("--prognn_dataset", action="store_true")
    parser.add_argument("--train_adj_only", action="store_true")
    parser.add_argument("--train_weight_only", action="store_true")
    parser.add_argument("--mix_optimizer", action="store_true")
    parser.add_argument("--only_prune_weight", action="store_true")
    parser.add_argument("--only_prune_adj", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--select_attack", action="store_true")
    parser.add_argument("--save_lottery_graph", action="store_true")  

    parser.add_argument("--random_sampling", action="store_true")
    parser.add_argument("--feature_sampling", action="store_true")

    parser.add_argument("--adj_sparsity", type=float, default=-1, help="adj_sparsity")
    parser.add_argument("--weight_sparsity", type=float, default=-1, help="weight_sparsity")
    parser.add_argument("--dataset_p", type=str,default="cora")
    parser.add_argument("--ptb_rate", type=float, default=0.05, help="noise ptb_rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")  # default:42
    args = parser.parse_args()


    if args.debug:
        print(args)

    if args.select_attack:
        if args.ptb_rate==0.05:
            datapath="../data/Arxiv_Graph/513_Lottery_arxiv_1.pt"
        elif args.ptb_rate==0.1:
            datapath="../data/Arxiv_Graph/525_Lottery_arxiv_2.pt"
        elif args.ptb_rate==0.15:
            datapath="../data/Arxiv_Graph/536_Lottery_arxiv_3.pt"
        elif args.ptb_rate==0.2:
            datapath="../data/Arxiv_Graph/548_Lottery_arxiv_4.pt"
        elif args.ptb_rate==0.25:
            datapath="../data/Arxiv_Graph/560_Lottery_arxiv_5.pt"
        elif args.ptb_rate==0:
            datapath="../data/Arxiv_Graph/Lottery_arxiv_original_Graph.pt"
        else:
            print("error=================================")
        if args.random_sampling:

            if args.ptb_rate==0.05:
                datapath="../data/Arxiv_Graph/529_random_arxiv_1.pt"
            elif args.ptb_rate==0.1:
                datapath="../data/Arxiv_Graph/541_random_arxiv_2.pt"
            elif args.ptb_rate==0.15:
                datapath="../data/Arxiv_Graph/552_random_arxiv_3.pt"
                datapath="../data/Arxiv_Graph/564_random_arxiv_4.pt"
            elif args.ptb_rate==0.25:
                datapath="../data/Arxiv_Graph/576_random_arxiv_5.pt"
            elif args.ptb_rate==0:
                print("error=================================")
            else:
                print("error=================================")
        if args.feature_sampling:
            if args.ptb_rate==0.05:
                datapath="../data/featureadj/featureadj_1_9.5_2431174_2444915_4870481.pt"
            elif args.ptb_rate==0.1:
                datapath="../data/featureadj/featureadj_2_9.5_2546528_2444915_4985771.pt"
            elif args.ptb_rate==0.15:
                datapath="../data/featureadj/featureadj_3_9.5_2662150_2444915_5101365.pt"
            elif args.ptb_rate==0.2:
                datapath="../data/featureadj/featureadj_4_9.5_2777808_2444915_5216971.pt"
            elif args.ptb_rate==0.25:
                datapath="../data/featureadj/featureadj_5_9.5_2893852_2444915_5332945.pt"
            elif args.ptb_rate==0:
                print("error=================================")
            else:
                print("error=================================")


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device,device1,device2 = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu", f"cuda:{args.device+1}" if torch.cuda.is_available() else "cpu",f"cuda:{args.device+2}" if torch.cuda.is_available() else "cpu"
    device , device1, device2= torch.device(device),torch.device(device1),torch.device(device2)
    dataname=args.dataset
    dataset = PygNodePropPredDataset(name=dataname, transform=T.ToSparseTensor(), root="../data")
    data = dataset[0]
    oracle_adj=copy.deepcopy(data.adj_t.to_symmetric()).to("cpu")
    num_classes=dataset.num_classes

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].to(device)
    ####substitute with prognn dataset
    if args.prognn_dataset:
        pass


    else:

        if args.attack:

            adj=torch.load(datapath).to("cpu")

            row,col,val=adj.coo()
            adj.set_value_(torch.ones(row.size()))
            data.adj_t=adj

        else:
    ################preprocess

            data.adj_t = data.adj_t.to_symmetric()
            adj = data.adj_t.to("cpu")
            row,col,val=adj.coo()


            adj.set_value_(torch.ones(row.size()))

    data = data.to(device)
    adj.to(device)



    K_order_graph=adj

    K_row,K_col,K_val=K_order_graph.coo()
    K_edge_index=torch.stack((K_col,K_row)).to(device)
    data.adj_t=K_edge_index

###################################



    evaluator = Evaluator(name=dataname)
    logger_0 = Logger(args.runs, args)
    logger_mask = Logger(args.runs, args)
    logger = Logger(args.runs, args)
    for run in range(args.runs):
        if args.use_sage:
            model = SAGE(data.num_features, args.hidden_channels, num_classes, args.num_layers, args.dropout).to(device)
        elif args.gcn_masked:
        
            model = GCN_Masked(data.num_features, args.hidden_channels, num_classes, args.num_layers, args.dropout).to(device)
        else:
            model = GCN(data.num_features, args.hidden_channels, num_classes, args.num_layers, args.dropout).to(device)
        best_testacc = 0
        best_model_weight = None
        best_adj_mask = None
        best_weight_mask = None
        best_valacc=0
        model.reset_parameters()
        rewind = copy.deepcopy({k: v for k, v in model.state_dict().items() if "mask" not in k})
        optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if "mask" not in n and p.requires_grad], lr=args.lr)

        
        estimator = EstimateAdj(K_edge_index).to(device)

        ###finetune get model weight
        for epoch in range(1, 1 + args.epoch1):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger_0.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            if valid_acc > best_valacc:
                best_testacc = test_acc
                best_valacc=valid_acc
                best_model_weight = copy.deepcopy(model.state_dict())
                if args.debug:
                    print(f"Bestmodelsaved for phase 0, " f"Run: {run + 1:02d}, " f"Epoch: {epoch:02d}, " f"Loss: {loss:.4f}, " f"Train: {100 * train_acc:.2f}%, " f"Valid: {100 * valid_acc:.2f}% " f"Test: {100 * test_acc:.2f}%")
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                if args.debug:
                    print(f"Run: {run + 1:02d}, " f"Epoch: {epoch:02d}, " f"Loss: {loss:.4f}, " f"Train: {100 * train_acc:.2f}%, " f"Valid: {100 * valid_acc:.2f}% " f"Test: {100 * test_acc:.2f}%")
        #######train_mask##########################################
        if args.debug:
            print(f"Bestmodelsaved for phase 0, " f"Valid: {100 * best_valacc:.2f}% "  f"Test: {100 * best_testacc:.2f}%")
        best_valacc=0
        best_testacc=0
        model.load_state_dict(best_model_weight)#######load best model

        # if args.train_adj_only:
        #     pass
        # else:
        #     model.replace_layers_with_masked()######add maskedlinear

    
        model.to(device)

        if args.train_weight_only:
            adj_optimizer = None
            mask_optimizer = optim.Adam([p for n, p in model.named_parameters() if "mask" in n and p.requires_grad], lr=args.lr_adj)
        elif args.train_adj_only:
            adj_optimizer = optim.Adam(estimator.parameters(), lr=args.lr_adj)
            mask_optimizer = None
        else:
            if args.mix_optimizer:
                mask_optimizer=None
                adj_optimizer=optim.Adam([p for n, p in model.named_parameters() if "mask" in n and p.requires_grad] + [p for p in estimator.parameters()], lr=args.lr_adj,weight_decay=args.weight_decay)
            else:

                adj_optimizer = optim.Adam(estimator.parameters(), lr=args.lr_adj,weight_decay=args.weight_decay)
                mask_optimizer = optim.Adam([p for n, p in model.named_parameters() if "mask" in n and p.requires_grad], lr=args.lr_weight,weight_decay=args.weight_decay)

        for epoch in range(1, 1 + args.epoch2):
            loss = train_mask(model, data, train_idx, mask_optimizer, estimator, adj_optimizer, args.alpha, args.beta, args.gamma)
            result = test(model, data, split_idx, evaluator, estimator)
            logger_mask.add_result(run, result)
            if  valid_acc > best_valacc:
                best_testacc = test_acc
                best_valacc=valid_acc
                best_weight_mask = copy.deepcopy(model.state_dict())
                best_adj_mask = copy.deepcopy(estimator.state_dict())
                if args.debug:
                    print(f"Bestmasksaved for phase 1, " f"Run: {run + 1:02d}, " f"Epoch: {epoch:02d}, " f"Loss: {loss:.4f}, " f"Train: {100 * train_acc:.2f}%, " f"Valid: {100 * valid_acc:.2f}% " f"Test: {100 * test_acc:.2f}%")
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                if not args.train_adj_only:
                    zero=model.compute_zero_pct()
                    one=model.compute_one_pct()
                    half=model.compute_half_pct()
                    # print(model.convs[0].lin.mask.mask_scores)
                    density=len(estimator.forward()[1])/estimator.edge_index.size(1)
                    if args.debug:
                        print(f"density of matri:{100*density:.2f}%","  size=",estimator.forward()[0].shape)
                    # print(estimator.mask.mask_scores.grad)
                        print(f"MaskRun: {run + 1:02d}, " f"Epoch: {epoch:02d}, " f"Loss: {loss:.4f}, " f"Train: {100 * train_acc:.2f}%, " f"Valid: {100 * valid_acc:.2f}% " f"Test: {100 * test_acc:.2f}% \n"  f"Weightmask0: {100 * zero:.2f}% " f"Weightmask1: {100 * one:.2f}% "  f"Weightmask_half: {100 * half:.2f}%")
                else:
                    density=len(estimator.forward()[1])/estimator.edge_index.size(1)
                    if args.debug:
                        print(f"density of matri:{100*density:.2f}%","  size=",estimator.forward()[0].shape)
                    # print(estimator.mask.mask_scores.grad)
                        print(f"MaskRun: {run + 1:02d}, " f"Epoch: {epoch:02d}, " f"Loss: {loss:.4f}, " f"Train: {100 * train_acc:.2f}%, " f"Valid: {100 * valid_acc:.2f}% " f"Test: {100 * test_acc:.2f}% \n"  )
        if args.debug:
            logger_0.print_statistics(run)
            logger_mask.print_statistics(run)
        ###########retrain lottery###########################################
        from draw_retrain_ticket import draw_ticket_mask,init_mask_score
        if not args.train_adj_only:
            best_weight_mask.update(rewind)
            model.load_state_dict(best_weight_mask)
            
            if args.weight_sparsity<0:
                pass
            else:
                weight_sparsity=args.weight_sparsity

                ticket_mask = draw_ticket_mask(model, weight_sparsity)
                
                init_mask_score(model, ticket_mask)
        if not args.train_weight_only:
        #prune edge_index##################
            estimator.load_state_dict(best_adj_mask)
            edge_index,edge_weight=estimator.forward()
            if args.adj_sparsity<0:
                pass
            else:
                if args.debug:
                    print("before prune: size=", edge_index.shape)
                adj_sparsity=args.adj_sparsity
                # estimator.mask
                ################positive mask_scores
                module =  estimator.mask
                module.data = torch.sigmoid(module.data)
                ######Prune mask_scores##############
                parameters_to_prune = []

                parameters_to_prune.append((estimator, 'mask'))



                parameters_to_prune = tuple(parameters_to_prune)

                prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=adj_sparsity,
                        )
                        ################extract mask scores_mask
                mask_scores_mask_dict = {}
                model_dict = estimator.state_dict()
                for key in model_dict.keys():
                    if 'mask_mask' in key:
                        mask_scores_mask_dict[key] = model_dict[key]
                ###########Init mask _scores##########
                module = estimator.mask
                module_mask = 'mask_mask'
                mask = mask_scores_mask_dict[module_mask]
                module.data = mask
                edge_index,edge_weight=estimator.forward()

                if args.debug:
                    print("after prune: size=", edge_index.shape)

    ###################################

            foundlottery=SparseTensor.from_edge_index(edge_index)
            foundlottery.set_value_(edge_weight)
            foundlottery=foundlottery.to("cpu")
            if args.debug:
                print("before prune: size=", edge_index.shape)
            sign=str(edge_index.size(1))
            # delta=torch_sparse.add(foundlottery,oracle_adj)
            import time
            timesign=time.time()
            if args.save_lottery_graph:

                torch.save(foundlottery,f"../data/makingadj/nol0mask/arxiv_found_adj_{args.ptb_rate}_{sign}_{timesign}.pt")

            data.adj_t=edge_index ##set best graph


        best_valacc=0
        best_testacc=0

        optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if "mask" not in n and p.requires_grad], lr=args.lr)
        ###load original initialization


        for epoch in range(1, 1 + args.epoch3):
            loss = train(model, data, train_idx, optimizer,edge_weight)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)
            if test_acc > best_testacc:
                best_testacc = test_acc
                best_valacc=valid_acc
                best_model_weight = copy.deepcopy(model.state_dict())
                if args.debug:
                    print(f"Bestmodelsaved for phase Final, " f"Run: {run + 1:02d}, " f"Epoch: {epoch:02d}, " f"Loss: {loss:.4f}, " f"Train: {100 * train_acc:.2f}%, " f"Valid: {100 * valid_acc:.2f}% " f"Test: {100 * test_acc:.2f}%")
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                if args.debug:
                    print(f"ReRun: {run + 1:02d}, " f"Epoch: {epoch:02d}, " f"Loss: {loss:.4f}, " f"Train: {100 * train_acc:.2f}%, " f"Valid: {100 * valid_acc:.2f}% " f"Test: {100 * test_acc:.2f}%")

        model.load_state_dict(best_model_weight)
        if args.debug:
            logger_0.print_statistics(run)
            logger_mask.print_statistics(run)
            logger.print_statistics(run)
    if args.debug:

        logger_0.print_statistics()
        logger_mask.print_statistics()
    testresult=logger.print_test_results(args,timesign)




if __name__ == "__main__":
    main()
