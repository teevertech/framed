from django.urls import path

from api import views

urlpatterns = [
    path("panels/generate", views.generate_panel_view, name="generate-panel"),
    path("panels/random", views.random_panel_view, name="random-panel"),
    path("sequence/run", views.run_sequence, name="run-sequence"),
    path("models", views.list_models, name="list-models"),
]
