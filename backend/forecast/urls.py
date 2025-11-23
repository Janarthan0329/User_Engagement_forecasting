# backend/forecast/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("user-engagement/", views.user_engagement_check, name="user_engagement_check"),
    path("forecast/", views.forecast_from_csv, name="forecast_from_csv"),
]
