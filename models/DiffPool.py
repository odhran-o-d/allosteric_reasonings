import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


class GNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        normalize=False,
        add_loop=False,
        lin=True,
    ):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, "bn{}".format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Diff_Pool_Encoder(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(3, 64, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(3, 64, 64, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(
            3 * 64, 64, 64, lin=False
        )  # self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.gnn3_embed = GNN(
            3 * 64, 64, 64, lin=False
        )  # self.lin2 = torch.nn.Linear(64, 6)

    def forward(self, x, adj, mask=None):

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)  # , print(x.shape)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)  # , print(x.shape)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)  # , print(x.shape)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)  # , print(x.shape)

        x = self.gnn3_embed(x, adj)  # , print(x.shape)

        x = x.mean(dim=1)
        return x  # print(x.shape) #x= F.relu(self.lin1(x)) #x= self.lin2(x)                       #return F.log_softmax(x, dim=-1), l1+l2, e1+e2


class DistMult_Decoder(torch.nn.Module):
    def __init__(
        self, args=None, dropout=0.05,
    ):
        super(DistMult_Decoder, self).__init__()
        self.inp_drop = torch.nn.Dropout(dropout)
        # self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, protein_embedded, drug_embedded, rel_embedded):

        drug_embedded = self.inp_drop(drug_embedded)
        protein_embedded = self.inp_drop(protein_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        print(drug_embedded.shape)
        print(protein_embedded.shape)
        print(rel_embedded.shape)
        pred = torch.mm(drug_embedded * rel_embedded, protein_embedded.transpose(1, 0))

        return pred


class Encoder_Decoder(torch.nn.Module):
    def __init__(
        self,
        protein_encoder=Diff_Pool_Encoder(),
        drug_encoder=Diff_Pool_Encoder(),
        decoder=DistMult_Decoder(),
        num_relationships=2,
    ):
        super(Encoder_Decoder, self).__init__()

        self.protein_encoder = protein_encoder
        self.drug_encoder = drug_encoder
        self.decoder = decoder  # this is the thing to build

        self.emb_rel = torch.nn.Embedding(
            num_relationships, embedding_dim=64 * 3, padding_idx=0
        )

    def init(self):
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, rel, d_graph, d_adj, d_mask, p_graph, p_adj, p_mask):

        rel_embedded = self.emb_rel(rel)
        rel_embedded = rel_embedded.squeeze()

        drug_embedded = self.drug_encoder(d_graph, d_adj, d_mask)
        protein_embedded = self.protein_encoder(d_graph, d_adj, d_mask)

        prediction = self.decoder(protein_embedded, drug_embedded, rel_embedded)

        print(prediction)
        return prediction
