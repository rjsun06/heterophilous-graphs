#%%
from metrics import METRICS, neigborhood_emb_post, neigborhood_emb_pre, neigborhood_emb_sym
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
dataset_name = 'cornell'
#%%
dataset = Dataset(name=dataset_name,
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
# pre = neigborhood_emb_pre(torch.nn.functional.one_hot(labels).float(),graph)
# post = neigborhood_emb_post(torch.nn.functional.one_hot(labels).float(),graph)
# sym = neigborhood_emb_post(torch.nn.functional.one_hot(labels).float(),graph)
#%%
results = []
for name, metric in METRICS.items():
    if name not in ['edges', 'nodes', 'classes', 'avg_dgree', 'edge_homophily', 'node_homophily',
    'class_homophily', 'adjusted_homophily', 
    'aggregation_homophily', 
    # 'aggregation_homophily_modified', 
    'our_homophily']: 
        continue
    value = metric(labels, features, graph)  # Calculate the metric
    results.append({"Metric": name, dataset_name: value})  # Append to results
# print(results)
ret = pd.DataFrame(results)
print(ret)

#%%
V = graph.nodes()
n = graph.num_nodes()
gi = dgl.add_self_loop(graph)
total = 0

def add_self_loops_manual(graph):
    graph = graph.clone()  # Clone the graph to avoid modifying the original
    graph.add_edges(graph.nodes(), graph.nodes())  # Add self-loops
    return graph

def agg_sim(graph,labels):
    emb = dgl.ops.copy_u_mean(add_self_loops_manual(graph), torch.nn.functional.one_hot(labels).float())
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
