from django.contrib import admin
from .models import MyModel, EdgeData

# Register your models here.
admin.site.register(MyModel)
admin.site.register(EdgeData)