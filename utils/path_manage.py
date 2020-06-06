
from os import getcwd
from os.path import join
import pickle

def get_files():
    master_directory = getcwd()
    data_directory = join(master_directory, 'data')

    with open(join(data_directory,'triplets.pickle'), 'rb') as handle:
        data = pickle.load(handle)
    with open(join(data_directory,'triplet_lookup.pickle'), 'rb') as handle:
        lookup = pickle.load(handle)
    with open(join(data_directory,'ASD_dictionary.pickle'), 'rb') as handle:
        ASD_dictionary = pickle.load(handle)
    with open(join(data_directory,'Sparse_dictionary.pickle'), 'rb') as handle:
        BCE_dictionary = pickle.load(handle)
    with open(join(data_directory,'edge_list.pickle'), 'rb') as handle:
        Edge_list = pickle.load(handle)
    with open(join(data_directory,'edge_features.pickle'), 'rb') as handle:
        Edge_features = pickle.load(handle)
    with open(join(data_directory,'drug_graph_list.pickle'), 'rb') as handle:
        Drug_graph_list = pickle.load(handle)
    with open(join(data_directory,'protein_graph_list.pickle'), 'rb') as handle:
        Protein_graph_list = pickle.load(handle)

    return data, lookup, ASD_dictionary, BCE_dictionary, Edge_list, Edge_features, Drug_graph_list, Protein_graph_list