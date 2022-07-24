from django import forms
from numpy import require

class UploadFileForm(forms.Form):
    file = forms.FileField(required=False)