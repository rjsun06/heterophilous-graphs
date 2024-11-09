import math
import dgl
import dgl.sparse
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

def edge_homoliphy(labels,features,graph):
    u, v = graph.edges()

    yu,yv = labels[u], labels[v]
    
    # Count same-class edges
    n_hedges = (yu == yv).sum().item()

    n_edges = graph.num_edges()
    
    # Calculate homophily
    if n_edges == 0:
        return 0  
    return n_hedges / n_edges


def node_homoliphy(labels,features,graph):
    #%%
    V = graph.nodes()
    v,u = graph.out_edges(V)
    yu,yv = labels[u], labels[v]
    ret = ((yv==yu)/graph.out_degrees(v)).sum().item()/graph.num_nodes()
    return ret


def class_homoliphy(labels,features,graph):
    V = graph.nodes()
    d = graph.out_degrees(V)
    v,u = graph.out_edges(V)
    yu,yv = labels[u], labels[v]
    d_inclass = torch.bincount(v,weights=yu==yv)
    n = graph.num_nodes()
    Ck = torch.bincount(labels)
    C = labels.max().item()+1
    hk = torch.bincount(labels, weights=d_inclass) / torch.bincount(labels, weights=d)
    return torch.clamp(hk-(Ck / n),min=0).mean().item()

def compat_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[0,:], edge_index[1,:]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max()+1
    H = torch.zeros((c,c)).to(edge_index.device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat((src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(H.dtype), add_idx, out=H[k,:], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H
def class_homoliphy1(labels,features,graph):
    edge_index = torch.stack(graph.edges())
    label = labels.squeeze()
    c = label.max()+1
    H = compat_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k,k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c-1
    return val.item()
    
def adjusted_homoliphy(labels,features,graph):
    return dgl.homophily.adjusted_homophily(graph,labels)

def neigborhood_emb_sym(X,gi): #~D^{-1/2}~A~D^{-1/2}X
    degrees = gi.out_degrees().float()
    norm_coefs = 1 / dgl.ops.u_mul_v(gi, degrees, degrees) ** 0.5

    return dgl.ops.u_mul_e_sum(gi, X, norm_coefs)

def neigborhood_emb_pre(X,gi): #~A~D^{-1}X
    return dgl.ops.copy_u_sum( gi, X/gi.out_degrees().view([-1,1]))

def neigborhood_emb_post(X,graph): #~D^{-1}~AX
    return dgl.ops.copy_u_mean( graph, X)

def modified_aggregation_homophily(labels,features,graph):
    V = graph.nodes()
    n = graph.num_nodes()
    gi = dgl.add_self_loop(graph)
    emb = neigborhood_emb_post(torch.nn.functional.one_hot(labels),graph)
    # agg_sim = emb@emb.T
    # X = torch.nn.functional.one_hot(labels).float()
    # label_sim = X@X.T
    # homo = (agg_sim*label_sim).sum(axis=1) / Ck[labels]
    # heter = (agg_sim*(label_sim==0)).sum(axis=1)/ (n-Ck[labels])
    # return (homo>heter).mean()
    total = 0
    for v in tqdm(V):
        agg_sim = emb[v]@emb.T
        label_sim = labels==labels[v]
        homo = agg_sim[label_sim].mean().item()
        heter = agg_sim[label_sim==0].mean().item()
        total += homo>heter
    return max(0,(total/n)*2-1)

def modified_aggregation_homophily(labels,features,graph):
    V = graph.nodes()
    n = graph.num_nodes()
    gi = dgl.add_self_loop(graph)
    emb = neigborhood_emb_post(torch.nn.functional.one_hot(labels),graph)
    # agg_sim = emb@emb.T
    # X = torch.nn.functional.one_hot(labels).float()
    # label_sim = X@X.T
    # homo = (agg_sim*label_sim).sum(axis=1) / Ck[labels]
    # heter = (agg_sim*(label_sim==0)).sum(axis=1)/ (n-Ck[labels])
    # return (homo>heter).mean()
    total = 0
    for v in tqdm(V):
        agg_sim = emb[v]@emb.T
        label_sim = labels==labels[v]
        homo = agg_sim[label_sim].mean().item()
        heter = agg_sim[label_sim==0].mean().item()
        total += homo>heter
    return max(0,(total/n)*2-1)

def aggregation_homophily_simp(labels,features,graph):
    V = graph.nodes()
    n = graph.num_nodes()
    emb = neigborhood_emb_post(torch.nn.functional.one_hot(labels).float(),graph)
    # agg_sim = emb@emb.T
    # X = torch.nn.functional.one_hot(labels).float()
    # label_sim = X@X.T
    # homo = (agg_sim*label_sim).sum(axis=1) / Ck[labels]
    # heter = (agg_sim*(label_sim==0)).sum(axis=1)/ (n-Ck[labels])
    # return (homo>heter).mean()
    total = 0
    for v in tqdm(V):
        agg_sim = emb[v]@emb.T
        label_sim = (labels==labels[v]).int()
        total += ((agg_sim - label_sim)**2).mean()
    return 1 - torch.sqrt(total/n).item()
    
METRICS = {
    'edges' : lambda labels,fetures,graph:
        graph.num_edges(),
    'nodes' : lambda labels,fetures,graph:
        graph.num_nodes(),
    'classes' : lambda labels,fetures,graph:labels.max().item()+1,
    'avg_dgree' : lambda labels,fetures,graph:
        graph.out_degrees().float().mean().item(),
    'edge_homophily': edge_homoliphy,
    'node_homophily': node_homoliphy,
    'class_homophily': class_homoliphy,
    # 'class_homophily1': class_homoliphy1,
    'adjusted_homophily': adjusted_homoliphy,
    'aggregation_homophily_m': modified_aggregation_homophily,
    'aggregation_homophily_m_-1': lambda labels,features,graph:modified_aggregation_homophily(labels,features,graph,k=-1),
    # 'aggregation_homophily_S': aggregation_homophily_simp,
}