from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views import View
#from sklearn.preprocessing import label_binarize
from .forms import UploadFileForm
from .models import MyModel, EdgeData
import pandas as pd
import json
from .tasks import build_model
import os
import csv
from io import StringIO

class upload_file(View):
    def get(self, request, *args, **kwargs):
        EdgeData.objects.all().delete()
        form = UploadFileForm()
        return render(request, 'linear_regression/progress.html', {'form': form})
    
    def post(self, request, *args, **kwargs):

        file = request.FILES['file'].read().decode('utf-8')
        node_to_label_dict, node_to_feature_dict, label_to_id_dict = read_data(file)
        task = build_model.delay(file, node_to_label_dict, node_to_feature_dict, label_to_id_dict)   

        return render(request, 'linear_regression/progress.html', {'task_id': task.task_id})


class result_view(View):
    def get(self, request, *args, **kwargs):
        
        fig = graph_function()
        gantt_plot = plot(fig, output_type="div")

        return render(request, 'linear_regression/result.html', {'plot_div': gantt_plot})



def read_data(file):
    csv_data = csv.reader(StringIO(file), delimiter=',')
    edges_read = False
    node_to_label_dict = dict()
    node_to_feature_dict = dict()
    label_to_id_dict = dict()

    row_cnt = 0
    for row in csv_data:
        row_cnt += 1
        #edge reading starts, line has form [node1_id, node2_id]
        if not edges_read:
            if row[0] == '#': #indicates last line of edge list
                edges_read = True
                continue
            
            _, created = EdgeData.objects.get_or_create(
                graph_id = 0,
                node1_id = row[0],
                node2_id =row[1],
                )            
        #label and feature reading starts, line has form [node_id, label, feature_1, feature_2, ..., feature_n]
        else:
            node_id, label, features = row[0], row[-1], row[1:-1]
            if label not in label_to_id_dict:
                label_to_id_dict[label] = len(label_to_id_dict)
            
            node_to_label_dict[int(node_id)] = label_to_id_dict[label] #int does not work, the keys become str in the task..
            node_to_feature_dict[int(node_id)] = list(map(float,features))
    
    return node_to_label_dict, node_to_feature_dict, label_to_id_dict

import plotly.graph_objects as go
from plotly.offline import plot
import networkx as nx

def graph_function():
    #G = nx.random_geometric_graph(200, 0.125)
    edge_list = EdgeData.objects.all()
    edge_list = [(edge.node1_id, edge.node2_id) for edge in edge_list]

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
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
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
    
    return fig
