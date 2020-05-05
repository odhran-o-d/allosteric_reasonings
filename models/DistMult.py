import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DistMult(torch.nn.Module):
    def __init__(self, embedding_dim=100, dropout=0.9, num_entities=2, num_relations=3):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(dropout)
        self.loss = torch.nn.CrossEntropyLoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1)
        # print(e1_embedded)
        rel_embedded = self.emb_rel(rel)
        # print(rel_embedded)
        e1_embedded = e1_embedded.squeeze()
        # print(e1_embedded)
        rel_embedded = rel_embedded.squeeze()
        # print(rel_embedded)

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        print(e1_embedded * rel_embedded)
        print(self.emb_e.weight.transpose(1, 0))

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        # print(pred)
        pred = torch.sigmoid(pred)

        return pred
