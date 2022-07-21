from django.db import models
from django.db.models.signals import post_delete
import os

class MyModel(models.Model):
    # file will be uploaded to MEDIA_ROOT/uploads
    file_field = models.FileField(upload_to='uploads/')
    #file = models.FileField()

class EdgeData(models.Model):

    graph_id = models.IntegerField()
    node1_id = models.IntegerField()
    node2_id = models.IntegerField()

