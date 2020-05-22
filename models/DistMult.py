import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DistMult(torch.nn.Module):
    def __init__(
        self,
        args=None,
        embedding_dim=100,
        dropout=0.05,
        num_entities=2,
        num_relations=3,
    ):
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

        # print(e1_embedded * rel_embedded)
        # print(self.emb_e.weight.transpose(1, 0))

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        # output should have a high dot product with the corrrect tail.
        # it's the matrix multiplication of all possible tails
        # dot product with every embedding. So you get a score for all entities.

        # print(pred)
        # pred = torch.sigmoid(pred)
        # the sigmoid is tecnically unecessary when not using BCE

        return pred


# when using cross entropy you use a batch of tripples.
# run it over cross entropy giving it the correct tail.
# if you're using cross entropy, sigmoid in the dist mult is unecessary

# BCE lets you formulate a loss based on 0-1 embeddings

if __name__ == "__main__":
    model = DistMult()
    for param in model.parameters():
        print(type(param.data), param.size())
