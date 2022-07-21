from django.shortcuts import render
from django.views import View
# Create your views here.

class landing(View):
    def get(self, request):
        return render(request, 'landing/index.html')

class about(View):
    def get(self, request):
        return render(request, 'landing/about.html')