from django.urls import path
from .views import (
    ChallengeView, ImageDetectionView, DocumentDetectionView,
    VideoLinkDetectionView, RobustnessTestView,
    FeedbackView, RetrainView, RetrainStatusView,
)

urlpatterns = [
    path('live/challenge/',       ChallengeView.as_view(),          name='live_challenge'),
    path('detect/',               ImageDetectionView.as_view(),     name='detect_image'),
    path('detect/document/',      DocumentDetectionView.as_view(),  name='detect_document'),
    path('detect/video/',         VideoLinkDetectionView.as_view(), name='detect_video'),
    path('detect/robustness/',    RobustnessTestView.as_view(),     name='detect_robustness'),
    path('feedback/',             FeedbackView.as_view(),           name='feedback'),
    path('retrain/',              RetrainView.as_view(),            name='retrain'),
    path('retrain/status/',       RetrainStatusView.as_view(),      name='retrain_status'),
]
