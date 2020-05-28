import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding_Encoder(torch.nn.Module):
    def __init__(
        self, embedding_dim=100, num_entities=2, num_relations=3,
    ):
        super(Embedding_Encoder, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)

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

        return e1_embedded, rel_embedded


class DistMult_Decoder(torch.nn.Module):
    def __init__(
        self, Encoder_Model=None, args=None, dropout=0.05,
    ):
        super(DistMult_Decoder, self).__init__()
        self.encodings = Encoder_Model
        self.inp_drop = torch.nn.Dropout(dropout)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, e1, rel):

        e1_embedded, rel_embedded = self.encodings("parameters go here")

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))

        return pred


if __name__ == "__main__":
    model = DistMult()
    for param in model.parameters():
        print(type(param.data), param.size())
