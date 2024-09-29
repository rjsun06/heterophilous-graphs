



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


METRICS = {
    'edges' : lambda labels,fetures,graph:
        graph.num_edges(),
    'nodes' : lambda labels,fetures,graph:
        graph.num_nodes(),
    'avg_dgree' : lambda labels,fetures,graph:
        graph.out_degrees().float().mean().item(),
    'edge_homophily': edge_homoliphy,
}