from django.urls import path
from .views import ChallengeView, ImageDetectionView, DocumentDetectionView, VideoLinkDetectionView, VoiceDetectionView

urlpatterns = [
    path('live/challenge/',  ChallengeView.as_view(),          name='live_challenge'),
    path('detect/',          ImageDetectionView.as_view(),     name='detect_image'),
    path('detect/document/', DocumentDetectionView.as_view(),  name='detect_document'),
    path('detect/video/',    VideoLinkDetectionView.as_view(), name='detect_video'),
    path('detect/voice/',    VoiceDetectionView.as_view(),     name='detect_voice'),
]
