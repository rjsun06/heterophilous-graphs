#%%
import math
import dgl
import dgl.sparse
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

def e(graph):
    adj = graph.adj().to_dense()
    def foo(fr,to=None):
        if to == None:to=graph.nodes()
        return adj[torch.meshgrid(fr,to)]
    return foo

def h_label(labels):
    def foo(fr,to=None):
        if to == None:to=range(labels.shape[0])
        return (labels[fr].unsqueeze(1) == labels[to]).float()
    return foo

def h_feature(f):
    f = f/torch.sqrt((f**2).sum(dim=1,keepdim=True))
    def foo(fr,to=None):
        if to == None:to=range(f.shape[0])
        return f[fr] @ f[to].T
    return foo

def h_rev(h):
    return lambda *arg: 1 - h(*arg)

def H(h,hh=None,batch=1):
    def foo0(*args):
        return h(*args).mean()
    if not hh: return foo0
    def foo(fr,to=None):
        if batch == 1:
            base = hh(fr,to)
            return (h(fr,to) * base).sum() / base.sum()

        a=torch.tensor(0.0).to(fr.device)
        b=torch.tensor(0.0).to(fr.device)
        for ffr in tqdm(fr.chunk(batch)):
            base = hh(ffr,to)
            a+=(h(ffr,to) * base).sum() 
            b+=base.sum()
        return a/b
    return foo

def Hc(h,hh):
    H1 = H(h,hh)
    H0 = H(h)
    def foo(*args):
        return H1(*args) - H0(*args)
    return foo

def Hcplus(h,hh):
    H0 = Hc(h,hh)
    def foo(*args):
        return max(H0(*args),0)
    return foo

def Hj(h,hh):
    H1 = H(h,hh)
    H0 = H(h,h_rev(hh))
    def foo(*args):
        return H1(*args) > H0(*args)
    return foo

def avg(domain,f):
    return sum(map(f,tqdm(domain)))/len(domain)

def avg1(domain,f):
    return sum(map(f,tqdm(domain)))/(len(domain)-1)

def div_node(graph):
    return graph.nodes().reshape(-1,1)

def div_class(labels):
    return [torch.where(labels==i) for i in torch.unique(labels)]

def pred_edge_homoliphy(labels,features,graph):
    return H(h_feature(features),e(graph),batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()).item()

def edge_homoliphy(labels,features,graph):
    return H(h_label(labels),e(graph),batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()).item()

def node_homoliphy(labels,features,graph):
    return avg(div_node(graph), H(h_label(labels),e(graph))).item()

def class_homoliphy(labels,features,graph):
    H = Hcplus(h_label(labels),e(graph))
    return avg1(div_class(labels),H).item()

def neigborhood_emb_sym(X,gi): #~D^{-1/2}~A~D^{-1/2}X
    degrees = gi.out_degrees().float()
    norm_coefs = 1 / dgl.ops.u_mul_v(gi, degrees, degrees) ** 0.5

    return dgl.ops.u_mul_e_sum(gi, X, norm_coefs)

def neigborhood_emb_pre(X,gi): #~A~D^{-1}X
    return dgl.ops.copy_u_sum( gi, X/gi.out_degrees().view([-1,1]))

def neigborhood_emb_post(X,gi): #~D^{-1}~AX
    return dgl.ops.copy_u_mean( gi, X)

def aggregation_homophily(labels,features,graph):
    gi = dgl.add_self_loop(graph)
    emb = neigborhood_emb_post(torch.nn.functional.one_hot(labels).float(),gi)
    H = Hc(h_feature(emb),h_label(labels))
    return avg(div_node(graph),lambda v: H(v)>=0).item()

def pred_our_naive(labels,features,graph):
    gi = dgl.add_self_loop(graph)
    emb = neigborhood_emb_post(features,gi)
    return H(h_feature(features),h_feature(emb),batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()).item()

def our_naive(labels,features,graph):
    gi = dgl.add_self_loop(graph)
    emb = neigborhood_emb_post(torch.nn.functional.one_hot(labels).float(),gi)
    return H(h_label(labels),h_feature(emb),batch=max(1,graph.num_nodes()**2//(10**7)))(graph.nodes()).item()

def our_homophily(labels,features,graph):
    gi = dgl.add_self_loop(graph)
    emb = neigborhood_emb_post(torch.nn.functional.one_hot(labels).float(),gi)
    H = Hc(h_label(labels),h_feature(emb))
    return avg(div_node(graph),H).item()

def our_class_homoliphy(labels,features,graph):
    classes,counts= torch.unique(labels,return_counts=True)
    gi = dgl.add_self_loop(graph)
    emb = neigborhood_emb_post(torch.nn.functional.one_hot(labels).float(),gi)
    H = Hcplus(h_label(labels),h_feature(emb))
    return avg1(div_class(labels),H).item()

def aggregation_homophily_simp(labels,features,graph):
    V = graph.nodes()
    n = graph.num_nodes()
    emb = neigborhood_emb_post(torch.nn.functional.one_hot(labels).float(),graph)
    total = 0
    for v in tqdm(V):
        agg_sim = emb[v]@emb.T
        label_sim = (labels==labels[v]).int()
        total += ((agg_sim - label_sim)**2).mean()
    return torch.sqrt(total/n).item()
    
METRICS_f = {
    'edges' : lambda labels,fetures,graph:
        graph.num_edges(),
    'nodes' : lambda labels,fetures,graph:
        graph.num_nodes(),
    'classes' : lambda labels,fetures,graph:labels.max().item()+1,
    'avg_dgree' : lambda labels,fetures,graph:
        graph.out_degrees().float().mean().item(),
    'edge_homophily': edge_homoliphy,
    # 'pred_edge_homophily': pred_edge_homoliphy,
    'node_homophily': node_homoliphy,
    'class_homophily': class_homoliphy,
    # 'class_homophily1': class_homoliphy1,
    # 'adjusted_homophily': adjusted_homoliphy,
    'aggregation_homophily': aggregation_homophily,
    'our_naive': our_naive,
    # 'pred_our_naive': pred_our_naive,
    'our_homophily': our_homophily,
    'our_class_homophily': our_class_homoliphy,
    # 'aggregation_homophily_modified': modified_aggregation_homophily,
    # 'aggregation_homophily_m_-1': lambda labels,features,graph:modified_aggregation_homophily(labels,features,graph,k=-1),
    'aggregation_homophily_S': aggregation_homophily_simp,
}