from cProfile import label
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views import View
from .forms import UploadFileForm
from .models import MyModel, EdgeData, NodeData, TestResult, AccuracyResults
import pandas as pd
import json
from .tasks import build_model
import os
import csv
from io import StringIO
from collections import OrderedDict

class upload_file(View):
    def get(self, request, *args, **kwargs):
        EdgeData.objects.all().delete()
        NodeData.objects.all().delete()
        TestResult.objects.all().delete()
        AccuracyResults.objects.all().delete()

        form = UploadFileForm()
        return render(request, 'linear_regression/download.html', {'form': form})
    
    def post(self, request, *args, **kwargs):
            
        try:
            file = request.FILES['file'].read().decode('utf-8')
        except:
            return render(request, 'linear_regression/no_file.html')
            
        node_to_label_dict, node_to_feature_dict, label_to_id_dict = read_data(file)
        train_acc, val_acc, test_acc, label_res_dict, counts = build_model(file, node_to_label_dict, node_to_feature_dict, label_to_id_dict)   

        TestResult.objects.bulk_create([TestResult(label=label, correct_count=counts[0], total_count=counts[1]) for label, counts in label_res_dict.items()])
        res = AccuracyResults(train=train_acc, val=val_acc, test=test_acc, train_cnt=counts[0], val_cnt=counts[1], test_cnt=counts[2])
        res.save()

        return render(request, 'linear_regression/download.html', {'result_ready': True})


class result_view(View):
    def get(self, request, *args, **kwargs):
        
        test_result = {res.label: [res.correct_count, res.total_count] for res in TestResult.objects.all()}
        acc = AccuracyResults.objects.all()[0]
        train_acc, val_acc, test_acc, train_cnt, val_cnt, test_cnt = acc.train, acc.val, acc.test, acc.train_cnt, acc.val_cnt, acc.test_cnt
        
        fig, label_distribution = graph_function()
        gantt_plot = plot(fig, output_type="div")

        test_result = OrderedDict(sorted(test_result.items()))
        label_distribution = OrderedDict(sorted(label_distribution.items()))

        return render(request, 'linear_regression/result.html', {'plot_div': gantt_plot,
                                                                'train_acc': train_acc*100,
                                                                'val_acc': val_acc*100,
                                                                'test_acc': test_acc*100,
                                                                'train_cnt':train_cnt,
                                                                'val_cnt':val_cnt,
                                                                'test_cnt':test_cnt,
                                                                'label_dist': label_distribution,
                                                                'test_result': test_result})

def read_data(file):
    csv_data = csv.reader(StringIO(file), delimiter=',')
    edges_read = False
    node_to_label_dict = dict()
    node_to_feature_dict = dict()
    label_to_id_dict = dict()

    import time
    start_time = time.time()

    db_edges = []

    for row in csv_data:
        #edge reading starts, line has form [node1_id, node2_id]
        if not edges_read:
            if row[0] == '#': #indicates last line of edge list
                edges_read = True
                print('edge reading:', time.time()-start_time)
                start_time = time.time()
                continue
            
            db_edges.append(EdgeData(graph_id = 0, node1_id = row[0], node2_id =row[1]))
                        
        #label and feature reading starts, line has form [node_id, label, feature_1, feature_2, ..., feature_n]
        else:
            node_id, label, features = row[0], row[-1], row[1:-1]
            if label not in label_to_id_dict:
                label_to_id_dict[label] = len(label_to_id_dict)
            
            node_to_label_dict[int(node_id)] = label_to_id_dict[label] #int does not work, the keys become str in the task..
            node_to_feature_dict[int(node_id)] = list(map(float,features))

    NodeData.objects.bulk_create([NodeData(node_id=node_id, label=label) for node_id, label in node_to_label_dict.items()])
    EdgeData.objects.bulk_create(db_edges)
    print('db update', time.time()-start_time)

    return node_to_label_dict, node_to_feature_dict, label_to_id_dict

import plotly.graph_objects as go
from plotly.offline import plot
import networkx as nx

def graph_function():
    #G = nx.random_geometric_graph(200, 0.125)
    edge_list = EdgeData.objects.all()
    node_list = NodeData.objects.all()
    edge_list = [(edge.node1_id, edge.node2_id) for edge in edge_list]
    node_label_dict = {node.node_id:node.label for node in node_list}

    G = nx.Graph()
    G.add_edges_from(edge_list)
    #nodePos = nx.circular_layout(G)
    nodePos = nx.random_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = nodePos[edge[0]]
        x1, y1 = nodePos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = nodePos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale='YlGnBu',
            reversescale=True,
            color=[node_label_dict[node] for node in G.nodes()],
            size=10,
            line_width=2))

    node_adjacencies = []
    node_text = []
    node_list = list(G)

    for i, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_class = node_label_dict[node_list[i]]
        node_text.append('Class: {}, Node degree:{}'.format(node_class, len(adjacencies[1])))

    #node_trace.marker.color = node_adjacencies
    node_trace.marker.size = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br> GRAPH PLOT',
                titlefont_size=20,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    label_distribution = dict()
    for label in node_label_dict.values():
        if label not in label_distribution:
            label_distribution[label] = 1
        else:
            label_distribution[label] += 1

    return fig, label_distribution
