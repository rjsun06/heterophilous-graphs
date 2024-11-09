
#%%
from metrics import METRICS
from datasets import Dataset
import pandas as pd
import dgl
import dgl.sparse
import numpy as np
import torch

#%%
available_datasets = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
'squirrel', 'squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed',
'chameleon', 'chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed',
'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin']
#%%
device = 'cuda:0'
#%%
dataset = Dataset(name='cornell',
                add_self_loops=False,
                device=device,
                use_sgc_features=False,
                use_identity_features=False,
                use_adjacency_features=False,
                do_not_use_original_features=False,
                bin2float=False)
labels = dataset.labels
features = dataset.node_features
graph = dataset.graph
#%%
V = graph.nodes()
n = graph.num_nodes()
gi = dgl.add_self_loop(graph)
total = 0
def agg_sim(graph,labels):
    emb = dgl.ops.copy_u_mean(dgl.ops.add_self_loops(graph), torch.nn.functional.one_hot(labels).float())
    def f(v):
        return emb[v]@emb.T
    return f

def homo_heter(fsim,labels):
    def f(v):
        sim = fsim(v)
        label_sim = labels==labels[v]
        inclass_sim = sim[label_sim]
        crossclass_sim = sim[label_sim==0]
        homo = inclass_sim.mean().item()
        heter = crossclass_sim.mean().item()
        return homo,heter
    return f

#%%
homo,heter = zip(*map(homo_heter(agg_sim(graph,labels),labels),V))
homo = torch.tensor(homo).to(device)
heter = torch.tensor(heter).to(device)
#%%
class_homo=torch.bincount(labels,weights=homo)/torch.bincount(labels)
class_heter=torch.bincount(labels,weights=heter)/torch.bincount(labels)
#%%
