import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv


class rgcn_link_predict(torch.nn.Module):
    def __init__(self, graph, hidden_dim=100):
        super(rgcn_link_predict, self).__init__()
        # Should add embedding layer at start  ----   self.emb_e = torch.nn.Embedding(data.num_nodes, hidden_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(
            graph.num_edge_features + 1, hidden_dim, padding_idx=0
        )

        self.conv1 = RGCNConv(
            in_channels=graph.num_nodes,
            out_channels=hidden_dim,
            num_relations=graph.num_edge_features,
            num_bases=30,
        )
        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=graph.num_edge_features,
            num_bases=30,
        )

    def forward(self, edge_index, edge_type):
        # Should add embedding layer to x? to do that you'll need to pass nodes to the forward, which is not done atm.
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        rel_embedded = self.emb_rel(edge_type)

        return x, rel_embedded

    def distmult(self, out, triplets):
        obj = out[0][triplets[:, 0]]
        rel = out[1][
            range(len(triplets))
        ]  # use rel = out[1][triplets[0:10,0]] or something similar when your code actually works
        sub = out[0][triplets[:, 2]]
        all_scores = torch.mm(
            (obj * rel), out[0].transpose(0, 1)
        )  # print(obj, rel, sub)
        return all_scores

