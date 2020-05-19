import numpy as np
import torch


def MRR(predictions, targets):

    top_k_objects = torch.topk(predictions, len(predictions[0]))
    placements = (top_k_objects[1] == targets).nonzero()
    ranks = placements[:, 1] + 1

    recporical_ranks = torch.ones(len(ranks)) / ranks
    mean_recporical_rank = torch.sum(recporical_ranks) / len(ranks)

    return mean_recporical_rank


def SRR(predictions, targets):

    top_k_objects = torch.topk(predictions, len(predictions[0]))
    placements = (top_k_objects[1] == targets).nonzero()
    ranks = placements[:, 1] + 1

    recporical_ranks = torch.ones(len(ranks)) / ranks
    sum_recporical_rank = torch.sum(recporical_ranks)

    return sum_recporical_rank


def ROC(x):
    return x


# this does not currently include masking!!!!
# However I don't think I need it because of my formulation right now?
