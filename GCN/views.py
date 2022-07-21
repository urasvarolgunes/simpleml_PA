from django.shortcuts import render
from django.views import View


class gcn_view(View):

    def get(self, request):
        return render(request, 'GCN/base.html')