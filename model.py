import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP


class Encoder_Net(nn.Module):
    def __init__(self, layers, dims):
        super(Encoder_Net, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])

    def forward(self, x):
        out1 = self.layers1(x)
        out2 = self.layers2(x)

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2


### <--- [MODIFIED] ---------------------------------------
class GCNEncoder_Net(nn.Module):
    """
    Optional GCN backbone for CCGC.
    Keep this class independent so default Encoder_Net (MLP) behavior is unchanged.
    """
    def __init__(self, layers, dims, dropout=0.0):
        super(GCNEncoder_Net, self).__init__()
        # Two contrastive branches, mirroring Encoder_Net's dual-head design.
        self.gc1 = GraphConvolution(dims[0], dims[1], dropout=dropout, act=lambda x: x)
        self.gc2 = GraphConvolution(dims[0], dims[1], dropout=dropout, act=lambda x: x)

    def forward(self, x, adj):
        out1 = self.gc1(x, adj)
        out2 = self.gc2(x, adj)

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2
### ---------------------------------------

