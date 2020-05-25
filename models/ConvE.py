import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from math import sqrt

from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# hidden size changs with embedding dimension!


class ConvE_args:
    def __init__(
        self,
        input_drop=0.05,
        hidden_drop=0.05,
        feat_drop=0.05,
        use_bias=True,
        embedding_shape1=20,
        hidden_size=9728,
    ):
        self.input_drop = input_drop
        self.hidden_drop = hidden_drop
        self.feat_drop = feat_drop
        self.use_bias = use_bias
        self.embedding_shape1 = embedding_shape1
        self.hidden_size = hidden_size


class ConvE(torch.nn.Module):
    def __init__(self, args, embedding_dim, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.loss = torch.nn.CrossEntropyLoss()

        self.emb_dim1 = int(sqrt(embedding_dim))
        self.emb_dim2 = self.emb_dim1

        # self.emb_dim1 = args.embedding_shape1
        # self.emb_dim2 = embedding_dim // self.emb_dim1
        # print(self.emb_dim2)
        self.hidden_size = 32 * ((2 * self.emb_dim1) - 2) * (self.emb_dim2 - 2)

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter("b", Parameter(torch.zeros(num_entities)))
        # self.fc = torch.nn.Linear(args.hidden_size, embedding_dim)
        self.fc = torch.nn.Linear(self.hidden_size, embedding_dim)
        # print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)

        x = self.inp_drop(stacked_inputs)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = x

        return pred
