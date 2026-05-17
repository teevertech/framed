from django.urls import path
from . import views

urlpatterns = [
    path("panels/generate", views.generate_panel),
    path("sequence/run", views.run_sequence),
    path("models", views.list_models),
]
