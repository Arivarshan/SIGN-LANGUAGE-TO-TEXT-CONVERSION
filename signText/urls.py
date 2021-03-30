from django.urls import path
from . import views

urlpatterns = [
     path('signText/', views.signToText),
]