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
import os, yaml, pandas, collections
base_dir = 'experiments'
def get_save_dir(dataset,name):
    idx = 1
    save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')
    return save_dir

def get_result(dataset,model):
    # print('get_result',dataset,model)
    save_dir = get_save_dir(dataset,model)
    try:
        with open(os.path.join(save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)
            metrics['acc'] = metrics.get('val accuracy mean') or metrics.get('val ROC AUC mean')
        df = pd.read_csv('our_naive.csv')[dataset]
        metrics['r_edge_homophily'] = df.iloc[0]
        metrics['r_our_naive'] = df.iloc[1]
        df = pd.read_csv('our_homo.csv')[dataset]
        metrics['r_node_homophily'] = df.iloc[0]
        metrics['r_our_homophily'] = df.iloc[1]
    except:
        return collections.defaultdict(lambda:0.0)
    return metrics

def get_measures(dataset):
    # print('get_result',dataset,model)
    save_dir = get_save_dir(dataset,model)
    try:
        with open(os.path.join(save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)
            metrics['acc'] = metrics.get('val accuracy mean') or metrics.get('val ROC AUC mean')
        df = pd.concat([
            pd.read_csv('our_naive.csv'),
            pd.read_csv('our_homo.csv'),
            pd.read_csv('metrics.csv'),
        ])
        return dict(zip(df['Metric'],df[dataset]))
    except:
        return collections.defaultdict(lambda:0.0)
    return metrics

print(get_result('tolokers','SAGE_l1'))
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
import matplotlib.pyplot as plt
import numpy as np



def c_model(x,model):
    if len(model)!=2:
        print(model)
        print('size=/=2')
        return
    a = [get_result(ds,model[0])[x] for ds in datasets]
    b = [get_result(ds,model[1])[x] for ds in datasets]
    fig, ax = plt.subplots()
    ax.set_xticklabels(datasets, rotation=60, ha='right', fontsize=10)
    ax2 = ax.twinx()
    ax2.set_ylabel('diff')
    
    ax.plot(datasets, a, 'bo', label=model[0])
    ax.plot(datasets, b, 'go', label=model[1])
    d = [i-j for i,j in zip(a,b)]
    ax2.bar(datasets, d, color=['green' if diff > 0 else 'red' for diff in d], alpha=0.6, label='Difference')

    ax.grid()
    ax.legend()
    return fig,ax,ax2

def c_x(x,model):
    if len(x)!=2:
        print(x)
        print('size=/=2')
        return
    a = [get_result(ds,model)[x[0]] for ds in datasets]
    b = [get_result(ds,model)[x[1]] for ds in datasets]
    fig, ax = plt.subplots()
    ax.set_xticklabels(datasets, rotation=60, ha='right', fontsize=10)
    ax2 = ax.twinx()
    ax2.set_ylabel('diff')
    
    ax.plot(datasets, a, 'bo', label=x[0])
    ax.plot(datasets, b, 'go', label=x[1])
    d = [i-j for i,j in zip(a,b)]
    ax2.bar(datasets, d, color=['green' if diff > 0 else 'red' for diff in d], alpha=0.6, label='Difference')

    ax.set_title(f"{x[0]} vs. {x[1]} for {model}")
    ax.grid()
    ax.legend()
    return fig,ax,ax2

def p_measure(x):
    a = [get_measures(ds)[x] for ds in datasets]
    fig, ax = plt.subplots()
    ax.set_xticklabels(datasets, rotation=60, ha='right', fontsize=10)
    ax.bar(datasets, a, label=x[0])
    ax.set_title(x)
    ax.grid()
    ax.legend()
    return fig,ax

def c_measure(x,method=lambda i,j:i-j,method_name='diff'):
    if len(x)!=2:
        print(x)
        print('size=/=2')
        return
    a = [get_measures(ds)[x[0]] for ds in datasets]
    b = [get_measures(ds)[x[1]] for ds in datasets]
    fig, ax = plt.subplots()
    ax.set_xticklabels(datasets, rotation=60, ha='right', fontsize=10)
    ax2 = ax.twinx()
    ax2.set_ylabel(method_name)
    
    ax.plot(datasets, a, 'bo', label=x[0])
    ax.plot(datasets, b, 'go', label=x[1])
    d = [method(i,j) for i,j in zip(a,b)]
    ax2.bar(datasets, d, color=['green' if diff > 0 else 'red' for diff in d], alpha=0.6, label='Difference')

    ax.set_title(f"{x[0]} vs. {x[1]}")
    ax.grid()
    ax.legend()
    return fig,ax,ax2

def graph_homo(model):
    eh = np.array([get_result(ds,model)['edge_homophily_avg'] for ds in datasets])
    reh = np.array([get_result(ds,model)['r_edge_homophily'] for ds in datasets])
    eeh = eh-reh

    naive = np.array([get_result(ds,model)['our_naive_avg'] for ds in datasets])
    rnaive = np.array([get_result(ds,model)['r_our_naive'] for ds in datasets])
    enaive = naive-rnaive

    fig, ax = plt.subplots()
    ax.set_xticklabels(datasets, rotation=60, ha='right', fontsize=10)
    ax.plot(datasets,eh) 
    ax.plot(datasets,reh) 
    ax.plot(datasets,naive) 
    ax.plot(datasets,rnaive)
    ax.bar(datasets, eeh, color=['yellow' if diff > 0 else 'yellow' for diff in eeh], alpha=0.6, label='Difference')
    ax.bar(datasets, enaive, color=['green' if diff > 0 else 'red' for diff in enaive], alpha=0.6, label='Difference')

    ax.grid()
    ax.legend()
    return fig,ax

model = 'GAT_l3'

# c_x(x='acc',model=('SAGE_l1','SAGE_l2'))
# c_x(x=('edge_homophily_avg','r_edge_homophily'),model=model)
c_x(x=('node_homophily_avg','r_node_homophily'),model=model)
# c_x(x=('our_naive_avg','r_our_naive'),model=model)
# c_x(x=('r_our_naive','r_edge_homophily'),model=model)
# c_x(x=('r_our_homophily','r_node_homophily'),model=model)
p_measure(x='avg_dgree')
# graph_homo('SAGE_l1')

