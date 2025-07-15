# urls.py
from django.urls import path
from .views import (
    UploadVoiceSampleView, GenerateVoiceModelView, TextToSpeechView,
    RegisterView, CustomTokenObtainPairView, CustomTokenRefreshView,
    UserDetailView, VoiceSampleListView, TrainingStatusView,VoiceSampleDeleteView,VoiceSampleStatsView
)

urlpatterns = [
    path('auth/register/', RegisterView.as_view(), name='auth_register'),
    path('auth/token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/user/', UserDetailView.as_view(), name='user-detail'),
    path('auth/token/refresh/', CustomTokenRefreshView.as_view(), name='token_refresh'),
    path('samples/stats/', VoiceSampleStatsView.as_view(), name='sample-stats'),
    path('upload-sample/', UploadVoiceSampleView.as_view(), name='upload-sample'),
    path('upload-sample/list/', VoiceSampleListView.as_view(), name='list-samples'),  # New
    path('upload-sample/<int:sample_id>/', VoiceSampleDeleteView.as_view(), name='delete-sample'),
    path('train-model/', GenerateVoiceModelView.as_view(), name='train-model'),
    path('train-model/<str:task_id>/', TrainingStatusView.as_view(), name='training-status'),  # New
    path('text-to-speech/', TextToSpeechView.as_view(), name='text-to-speech'),
]