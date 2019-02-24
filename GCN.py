import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from layer import GraphConvolution,Readout

class GCN(nn.Module):
    def __init__(self,nfeat,nnode,nhid,dropout):
        super(GCN,self).__init__()

        self.gc1 = GraphConvolution(nfeat,nhid)
        self.gc2 = GraphConvolution(nhid,nhid)
        self.dropout = dropout
        self.rd = Readout(nnode)
        self.fc1 = nn.Linear(nnode,1)
        nn.init.kaiming_normal_(self.fc1.weight)
    
    def forward(self,x,adj):
        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x,self.dropout,training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x,self.dropout,training=self.training)
        x = F.relu(self.rd(x))
        x = self.fc1(x)
        return x