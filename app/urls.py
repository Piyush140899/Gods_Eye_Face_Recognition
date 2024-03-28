from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('add_photos/', views.add_photos, name='add_photos'),
    path('add_photos/<slug:emp_id>/', views.click_photos, name='click_photos'),
    path('train_model/', views.train_model, name='train_model'),
    path('detected/', views.detected, name='detected'),
    path('identify/', views.identify, name='identify'),
    path('add_emp/', views.add_emp, name='add_emp'),
    path('prev_map/map', views.map, name='map'),
    path('prev_map/', views.prev_map, name='prev_map'),
    path('n_plate1/n_plate', views.n_plate, name='n_plate'),
    path('n_plate1/', views.n_plate1, name='n_plate1'),
    path('v_view/', views.v_view, name='v_view'),

]
