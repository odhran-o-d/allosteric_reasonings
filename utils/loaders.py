import torch
import numpy as np
from sklearn.utils import shuffle


def load_data(data, batch_number):
    shuffled_data = shuffle(data)
    objects, relationships, subjects = np.split(shuffled_data, 3, axis=1)
    batched = lambda x: np.array_split(x, batch_number)
    # if np.array_split isn't deterministic you may have different size splits!
    batched_objects = batched(objects)
    batched_subjects = batched(subjects)
    batched_relationships = batched(relationships)
    return batched_objects, batched_subjects, batched_relationships


def get_onehots(targets, vocab_len):
    batch_size = len(targets)
    y = torch.LongTensor(targets)
    y_onehot = torch.FloatTensor(batch_size, vocab_len)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot


def load_data_for_BCE(data, batch_number):
    shuffled_data = shuffle(data)
    objects, relationships, subjects = np.split(shuffled_data, 3, axis=1)
    batched = lambda x: np.array_split(x, batch_number)
    # if np.array_split isn't deterministic you may have different size splits!
    batched_objects = batched(objects)
    batched_subjects = batched(subjects)
    batched_relationships = batched(relationships)
    return batched_objects, batched_subjects, batched_relationships
