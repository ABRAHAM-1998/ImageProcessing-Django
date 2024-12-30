from django.urls import path
from .views import upload_and_process_image
from .views import generate_panorama_video,get_progress,trigger_panorama_generation
urlpatterns = [
    path('', upload_and_process_image, name='upload_image'),
    path('generate_panorama_video/', generate_panorama_video, name='generate_panorama_video'),
    path('get_progress/', get_progress, name='get_progress'),
    path('trigger_panorama_generation/', trigger_panorama_generation, name='trigger_panorama_generation'),
]