#%%
from metrics import METRICS, neigborhood_emb_post, neigborhood_emb_pre, neigborhood_emb_sym, nodewise_homophily
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
dataset_name = 'texas'
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
from metrics_framework import *
gi = dgl.add_self_loop(graph)
agg_labels = neigborhood_emb_sym(torch.nn.functional.one_hot(labels).float(),gi)
agg_features = neigborhood_emb_sym(features,gi)
h1 = h_label(labels)
gh1 = h_feature(features) 
h2 = h_feature(agg_labels) 
gh2 = h_feature(agg_features) 
print()
print(H(h1,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))
print(H(h1,h2,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))
print(H(h1,gh1,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))
print(H(h1,gh2,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))

print()
print(H(gh1,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))
print(H(gh1,h1,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))
print(H(gh1,h2,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))

print()
print(H(gh2,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))
print(H(gh2,h1,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))
print(H(gh2,h2,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))

print()
print(H(h2,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))
print(H(h2,gh2,batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()))


#%%
datasets = [
    'actor',
    'amazon-ratings',
    'chameleon',
    'chameleon-directed',
    'chameleon-filtered',
    'chameleon-filtered-directed',
    'cornell',
    'minesweeper',
    'questions',
    'roman-empire',
    'squirrel',
    'squirrel-directed',
    'squirrel-filtered',
    'squirrel-filtered-directed',
    'texas',
    'texas-4-classes',
    'tolokers',
    'wisconsin',
]
for ds in datasets:
    dataset = Dataset(name=ds,
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
    result = nodewise_homophily(labels,features,graph)
    torch.save(result,'metrics/'+ds+'-nodewise_homo.pt')
    result = nodewise_our_homophily(labels,features,graph)
    torch.save(result,'metrics/'+ds+'-nodewise_our_homo.pt')