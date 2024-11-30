#%%
import os, yaml, collections
import pandas as pd
base_dir = 'experiments'
def get_save_dir(dataset,name,idx=1):
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

model = 'FAGCN_l1'

# c_x(x='acc',model=('SAGE_l1','SAGE_l2'))
# c_x(x=('edge_homophily_avg','r_edge_homophily'),model=model)
c_x(x=('node_homophily_avg','r_node_homophily'),model=model)
# c_x(x=('our_naive_avg','r_our_naive'),model=model)
# c_x(x=('r_our_naive','r_edge_homophily'),model=model)
# c_x(x=('r_our_homophily','r_node_homophily'),model=model)
# p_measure(x='avg_dgree')
# graph_homo('SAGE_l1')

#%%
import torch
from metrics import f_nodewise_homophily,nodewise_our_homophily
from datasets import Dataset
def get_nw_homo(dataset,model,idx=1,homo='nodewise_homophily'):
    save_dir = get_save_dir(dataset,model,idx)
    return torch.load(os.path.join(save_dir,homo+'.pt')).cpu()

def get_f_nw_homo(dataset,model,idx=1,homo=f_nodewise_homophily):
    save_dir = get_save_dir(dataset,model,idx)
    pred = torch.load(os.path.join(save_dir,'last_pred.pt'))
    logits = torch.load(os.path.join(save_dir,'last_logits.pt'))
    graph=Dataset(name=dataset,
                add_self_loops=False,
                device='cuda:0',
                use_sgc_features=False,
                use_identity_features=False,
                use_adjacency_features=False,
                do_not_use_original_features=False,
                bin2float=False).graph
    return homo(pred,logits,graph).cpu()

def get_nw_homo_gt(dataset,homo='nodewise_homophily'):
    return torch.load(os.path.join('metrics',dataset+'-'+homo+'.pt')).cpu()

def graph_node_homo_dots(dataset,model,idx=1):

    res = get_f_nw_homo(dataset,model,idx,nodewise_our_homophily)
    gt = get_nw_homo_gt(dataset,'nodewise_our_homo')
    print(res.shape,gt.shape)

    fig, ax = plt.subplots()
    ax.scatter(gt,res)

    ax.grid()
    ax.legend()
    return fig,ax

graph_node_homo_dots('cornell','test',7)
# %%
