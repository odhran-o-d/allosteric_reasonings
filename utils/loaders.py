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


def load_evaluation_data(data, batch_number):
    shuffled_data = shuffle(data)
    objects, relationships, subjects = np.split(shuffled_data, 3, axis=1)
    batched = lambda x: np.array_split(x, batch_number)
    # if np.array_split isn't deterministic you may have different size splits!
    batched_objects = batched(objects)
    batched_subjects = batched(subjects)
    batched_relationships = batched(relationships)
    return batched_objects, batched_subjects, batched_relationships


def load_data_for_BCE(data, batch_number):
    shuffled_data = shuffle(data)
    objects, relationships, subjects = np.split(shuffled_data, 3, axis=1)
    batched = lambda x: np.array_split(x, batch_number)
    # if np.array_split isn't deterministic you may have different size splits!
    batched_objects = batched(objects)
    batched_subjects = batched(subjects)
    batched_relationships = batched(relationships)
    return batched_objects, batched_subjects, batched_relationships
