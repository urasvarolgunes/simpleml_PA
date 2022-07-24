import time
from linear_regression.models import MyModel, EdgeData
from simple_ml.settings import MEDIA_ROOT, MEDIA_URL
import os
import pandas as pd
from .models import EdgeData
from .gcn_models import GCN
#from .utils import normalize
from .train import train, test
import numpy as np
import torch
import csv
from io import StringIO
import scipy.sparse as sp

def build_model(file, node_to_label_dict, node_to_feature_dict, label_to_id_dict):

    n_nodes = len(node_to_label_dict)
    epochs = 1000
    print(node_to_label_dict)
    node_to_nodeid = {node:i for i,node in enumerate(node_to_label_dict.keys())}
    print(node_to_nodeid)
    nodeid_to_node = {i:node for i,node in enumerate(node_to_label_dict.keys())}
    print(nodeid_to_node)    
    edge_list = EdgeData.objects.all()

    features = torch.FloatTensor([node_to_feature_dict[nodeid_to_node[node_id]] for node_id in range(n_nodes)])
    features = torch.nn.functional.normalize(features, p=1, dim=0)
    labels = torch.LongTensor([node_to_label_dict[nodeid_to_node[node_id]] for node_id in range(n_nodes)])

    edges = np.array([(node_to_nodeid[edge.node1_id], node_to_nodeid[edge.node2_id]) for edge in edge_list])
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    idx_train = range(int(n_nodes*0.8))
    idx_val = range(int(n_nodes*0.8), int(n_nodes*0.9))
    idx_test = range(int(n_nodes*0.9), n_nodes)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=64,
                nclass=labels.max().item() + 1,
                dropout=0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    t_total = time.time()
    for epoch in range(epochs):
        train_acc, val_acc = train(epoch, model, optimizer, features, labels, adj, idx_train, idx_val)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('testing')
    test_acc = test(model, features, labels, adj, idx_test)
    
    return train_acc, val_acc, test_acc


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx