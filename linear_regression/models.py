from django.db import models
from django.db.models.signals import post_delete
import os

from linear_regression.train import train

class MyModel(models.Model):
    # file will be uploaded to MEDIA_ROOT/uploads
    file_field = models.FileField(upload_to='uploads/')
    #file = models.FileField()

class EdgeData(models.Model):

    graph_id = models.IntegerField()
    node1_id = models.IntegerField()
    node2_id = models.IntegerField()

class NodeData(models.Model):

    node_id = models.IntegerField()
    label = models.IntegerField()

class TestResult(models.Model):

    label = models.IntegerField()
    correct_count = models.IntegerField()
    total_count = models.IntegerField()

class AccuracyResults(models.Model):

    train = models.FloatField() 
    val = models.FloatField()
    test = models.FloatField()
    train_cnt = models.IntegerField()
    val_cnt = models.IntegerField()
    test_cnt = models.IntegerField()