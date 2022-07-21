from django.urls import path
from .views import gcn_view

app_name = 'GCN'

urlpatterns = [
    path('', gcn_view.as_view() , name='gcn_view'),
]