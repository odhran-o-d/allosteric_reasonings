import torch
import numpy as np


def load_data(data, batch_number):
    np.random.shuffle(data)
    objects, relationships, subjects = np.split(data, 3, axis=1)
    batched = lambda x: np.array_split(x, batch_number)
    # if np.array_split isn't deterministic you may have different size splits!
    batched_objects = batched(objects)
    batched_subjects = batched(subjects)
    batched_relationships = batched(relationships)
    return batched_objects, batched_subjects, batched_relationships
