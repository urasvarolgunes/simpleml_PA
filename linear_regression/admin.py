from django.contrib import admin
from .models import MyModel, EdgeData, NodeData, TestResult, AccuracyResults

# Register your models here.
admin.site.register(MyModel)
admin.site.register(EdgeData)
admin.site.register(NodeData)
admin.site.register(TestResult)
admin.site.register(AccuracyResults)