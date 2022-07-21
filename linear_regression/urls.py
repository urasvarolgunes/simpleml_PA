from django.urls import path
from linear_regression.views import upload_file, result_view
from . import views

app_name = 'linear_regression'

urlpatterns = [
    path('', upload_file.as_view() , name='demo'),
    path('result', result_view.as_view() , name='result'),
]