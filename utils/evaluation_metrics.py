import numpy as np
import torch
from sklearn import metrics


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


def auprc_auroc_ap(target_tensor, score_tensor):
    target_tensor = torch.flatten(target_tensor)
    score_tensor = torch.flatten(score_tensor)

    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()

    auroc = metrics.roc_auc_score(y, pred)
    ap = metrics.average_precision_score(y, pred)
    y, xx, _ = metrics.precision_recall_curve(y, pred)
    auprc = metrics.auc(xx, y)

    return auprc, auroc, ap
