from django.urls import path
from .views import landing, about

app_name = 'landing'

urlpatterns = [
    path('', landing.as_view() , name='landing'),
    path('about', about.as_view(), name='about')
]