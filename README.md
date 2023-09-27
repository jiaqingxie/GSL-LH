
#Regular-size datasets GSL-LH

In datafold Regular-size.
example: 'python train.py --gcn --seed 42 --dataset cora --attack meta --ptb_rate 0.25 \
 --weight_sparsity {weight_sparsity} --adj_sparsity {adj_sparsity} \
 --lr {lr} --lr_adj {lr_adj} --lr_mask {lr_mask} --seed {seed}'
other baseline model in test_{baselinemodel}.py

##Large-scale datasets GSL-LH
#'python FatherGraph.py' to make large graph sampling.
In datafold Large-arxiv.
example:  'python gnn.py --attack --select_attack --save_lottery_graph \
                            --ptb_rate {ptb_rate} --device {device} --gcn_masked \
                        --lr {lr} --lr_adj {lr_adj} --lr_weight {lr_weight} --seed {seed} \
                            --weight_sparsity {weight_sparsity} --adj_sparsity {adj_sparsity} --runs 5'

#Arxiv baselines

In datafold arxiv-baselines. From OGB.
example: 'python gnn.py'

#PPRGO-based methods

In datafold LTH_for_arxiv.
example: 
set SparseTensor adjacent matrix saving path=XXX.pt
to use learned graph in PPRGO, use
'python run.py --attacked_graph_path=XXX.pt'