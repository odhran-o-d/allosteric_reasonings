import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Complex(torch.nn.Module):
    def __init__(
        self, args, embedding_dim=100, dropout=0.05, num_entities=2, num_relations=4
    ):
        super(Complex, self).__init__()
        self.num_entities = num_entities

        self.emb_e_real = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=0)

        self.emb_rel_real = torch.nn.Embedding(
            num_relations, embedding_dim, padding_idx=0
        )
        self.emb_rel_img = torch.nn.Embedding(
            num_relations, embedding_dim, padding_idx=0
        )

        self.inp_drop = torch.nn.Dropout(dropout)
        # self.loss = torch.nn.BCELoss()
        self.loss = torch.nn.CrossEntropyLoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img = self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(
            e1_embedded_real * rel_embedded_real, self.emb_e_real.weight.transpose(1, 0)
        )
        realimgimg = torch.mm(
            e1_embedded_real * rel_embedded_img, self.emb_e_img.weight.transpose(1, 0)
        )
        imgrealimg = torch.mm(
            e1_embedded_img * rel_embedded_real, self.emb_e_img.weight.transpose(1, 0)
        )
        imgimgreal = torch.mm(
            e1_embedded_img * rel_embedded_img, self.emb_e_real.weight.transpose(1, 0)
        )
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        # pred = torch.sigmoid(pred)

        return pred
