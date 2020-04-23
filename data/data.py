import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch

from utils import add_self_loops, normalize_adj


class Data(object):
    def __init__(self, adj, edge_list, features, labels, adjmat):
        self.adj = adj
        self.edge_list = edge_list
        self.features = features
        self.labels = labels
        self.adjmat = adjmat
        self.num_nodes = features.size(0)
        self.num_features = features.size(1)
        self.num_classes = int(torch.max(labels)) + 1

    def to(self, device):
        self.adj = self.adj.to(device)
        self.edge_list = self.edge_list.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.adjmat = self.adjmat.to(device)


def load_data(dataset_str, ntrain=20, seed=None):
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        data = load_planetoid_data(dataset_str)
    elif dataset_str in ['karate']:
        data = load_karate_data()
    elif dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        data = load_geom_data(dataset_str, ntrain, seed)
    else:
        data = load_npz_data(dataset_str, ntrain, seed)
    return data


def load_planetoid_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("data/planetoid/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.Tensor(out))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1))
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    edge_list = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj(edge_list)
    adjmat = torch.FloatTensor(nx.to_numpy_matrix(G) + np.eye(features.size(0)))

    data = Data(adj, edge_list, features, labels, adjmat)

    return data


def load_karate_data():
    G = nx.karate_club_graph()
    N = G.number_of_nodes()
    features = torch.eye(N)
    label_map = {'Mr. Hi': 0, 'Officer': 1}
    labels = torch.tensor([label_map[G.nodes()[i]['club']] for i in range(N)])
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    edge_list = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    edge_list = add_self_loops(edge_list, G.number_of_nodes())
    adj = normalize_adj(edge_list)
    adjmat = torch.FloatTensor(nx.to_numpy_matrix(G) + np.eye(N))

    data = Data(adj, edge_list, features, labels, adjmat)

    return data


def load_npz_data(dataset_str, ntrain, seed):
    with np.load('data/npz/' + dataset_str + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)
    adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                            shape=loader['adj_shape']).tocoo()
    if dataset_str[:2] == 'ms':
        edge_list = torch.cat((torch.tensor(adj_mat.row).type(torch.int64).view(1, -1),
                               torch.tensor(adj_mat.col).type(torch.int64).view(1, -1)), dim=0)
    else:
        edge_list1 = torch.cat((torch.tensor(adj_mat.row).type(torch.int64).view(1, -1),
                                torch.tensor(adj_mat.col).type(torch.int64).view(1, -1)), dim=0)
        edge_list2 = torch.cat((torch.tensor(adj_mat.col).type(torch.int64).view(1, -1),
                                torch.tensor(adj_mat.row).type(torch.int64).view(1, -1)), dim=0)
        edge_list = torch.cat([edge_list1, edge_list2], dim=1)

    edge_list = add_self_loops(edge_list, loader['adj_shape'][0])
    adj = normalize_adj(edge_list)
    if 'attr_data' in loader:
        feature_mat = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                    shape=loader['attr_shape']).todense()
    elif 'attr_matrix' in loader:
        feature_mat = loader['attr_matrix']
    else:
        feature_mat = None
    features = torch.tensor(feature_mat)

    if 'labels_data' in loader:
        labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                               shape=loader['labels_shape']).todense()
    elif 'labels' in loader:
        labels = loader['labels']
    else:
        labels = None
    labels = torch.tensor(labels).long()

    data = Data(adj, edge_list, features, labels, None)
    return data


def load_geom_data(dataset_str, ntrain, seed):
    # Feature and Label preprocessing
    with open('data/geom_data/{}/out1_node_feature_label.txt'.format(dataset_str)) as f:
        feature_labels = f.readlines()
    feat_list = []
    label_list = []
    for fl in feature_labels[1:]:
        id, feat, lab = fl.split('\t')
        feat = list(map(int, feat.split(',')))
        feat_list.append(feat)
        label_list.append(int(lab))
    features = torch.FloatTensor(feat_list)
    labels = torch.tensor(label_list).long()

    # Graph preprocessing
    with open('data/geom_data/{}/out1_graph_edges.txt'.format(dataset_str)) as f:
        edges = f.readlines()
    edge_pairs = []
    G = nx.Graph()
    for e in edges[1:]:
        u, v = map(int, e.split('\t'))
        edge_pairs.append((u, v))
    G.add_edges_from(edge_pairs)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    edge_list = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj(edge_list)

    data = Data(adj, edge_list, features, labels, None)
    return data


def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def split_data(labels, n_train_per_class, n_val, seed):
    np.random.seed(seed)
    n_class = int(torch.max(labels)) + 1
    train_idx = np.array([], dtype=np.int64)
    remains = np.array([], dtype=np.int64)
    for c in range(n_class):
        candidate = torch.nonzero(labels == c).T.numpy()[0]
        np.random.shuffle(candidate)
        train_idx = np.concatenate([train_idx, candidate[:n_train_per_class]])
        remains = np.concatenate([remains, candidate[n_train_per_class:]])
    np.random.shuffle(remains)
    val_idx = remains[:n_val]
    test_idx = remains[n_val:]
    train_mask = index_to_mask(train_idx, labels.size(0))
    val_mask = index_to_mask(val_idx, labels.size(0))
    test_mask = index_to_mask(test_idx, labels.size(0))
    return train_mask, val_mask, test_mask