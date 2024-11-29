import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


class FAGCN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, graph, drop=0.6, eps=0.2, num_layer=3
    ):
        super(FAGCN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(p=drop)
        self.eps = eps
        self.num_layer = num_layer

        self.gnn_layer = nn.ModuleList()
        for _ in range(self.num_layer):
            self.gnn_layer.append(torch_geometric.nn.FAConv(self.hidden_size, drop))

        self.t1 = nn.Linear(self.input_size, self.hidden_size)
        self.t2 = nn.Linear(self.hidden_size, self.output_size)
        self.edge_index = graph

    def forward(self, graph, x):
        x = self.dropout(x)
        x = F.relu(self.t1(x))
        x = self.dropout(x)

        h_0 = x
        for i in range(len(self.gnn_layer)):
            x = self.gnn_layer[i](x,h_0,self.edge_index)

        x = self.t2(x).squeeze(1)
        return x