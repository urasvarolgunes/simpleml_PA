# Celery
#from asyncore import file_dispatcher
from celery import shared_task
# Celery-progress
from celery_progress.backend import ProgressRecorder
# Task imports
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
from django.db import transaction

# Celery Task
@shared_task(bind=True)
def build_model(self, file, node_to_label_dict, node_to_feature_dict, label_to_id_dict):
    
    epochs = 500
    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(0, epochs, description="Training Model...")
    print(node_to_label_dict)
    node_to_nodeid = {int(node):i for i,node in enumerate(node_to_label_dict.keys())}
    print(node_to_nodeid)
    nodeid_to_node = {i:int(node) for i,node in enumerate(node_to_label_dict.keys())}
    print(nodeid_to_node)    
    edge_list = EdgeData.objects.all()
    edge_list = [(node_to_nodeid[edge.node1_id], node_to_nodeid[edge.node2_id]) for edge in edge_list]
    #edge_list = [(node_to_nodeid[node1], node_to_nodeid[node2]) for node1, node2 in edge_list]
    n_nodes = len(node_to_label_dict)
    indices = torch.LongTensor(edge_list).T #need to transpose as sparse matrix expects shape [2, n_edges]
    values = torch.ones(indices.shape[1]).float()

    adj = torch.sparse.FloatTensor(indices, values, size=torch.Size([n_nodes, n_nodes]))
    features = torch.FloatTensor([node_to_feature_dict[str(nodeid_to_node[node_id])] for node_id in range(n_nodes)])
    labels = torch.LongTensor([node_to_label_dict[str(nodeid_to_node[node_id])] for node_id in range(n_nodes)])

    idx_train = range(int(n_nodes*0.8))
    idx_val = range(int(n_nodes*0.8), int(n_nodes*0.9))
    idx_test = range(int(n_nodes*0.9), n_nodes)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=32,
                nclass=labels.max().item() + 1,
                dropout=0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    t_total = time.time()
    for epoch in range(epochs):
        train(epoch, model, optimizer, features, labels, adj, idx_train, idx_val)
        progress_recorder.set_progress(epoch + 1, epochs, description="Training Model...")

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('testing')
    test(model, features, labels, adj, idx_test)
    
    #progress_recorder = ProgressRecorder(self)
    #progress_recorder.set_progress(1, 10, description="Training Model...")

    return 'Finished.'