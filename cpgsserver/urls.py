# Developed By Tecktrio At Liquidlab Infosystems
# Project: Routes 
# Version: 1.0
# Date: 2025-03-08
# Description: Endpoints/ routes for the server where devices may hit

import threading
from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView
from cpgsapp import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from cpgsapp.controllers.NetworkController import Broker

ThreadalreadyRunning = False

# Endpoints
urlpatterns = [
    path('admin/', admin.site.urls),
    path('network_handler', views.NetworkHandler.as_view()),
    path('account_handler', views.AccountHandler.as_view()),
    path('monitor_handler', views.MonitorHandler.as_view()),
    path('calibrate_handler', views.CalibrateHandler.as_view()),
    path('mode_handler', views.ModeHandler.as_view()),
    path('reboot', views.reboot),
    path('',TemplateView.as_view(template_name = 'index.html'))
] + staticfiles_urlpatterns()

if not ThreadalreadyRunning:
    ThreadalreadyRunning = True
    threading.Thread(target=views.ModeMonitor).start() 

