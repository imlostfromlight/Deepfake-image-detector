from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    def ready(self):
        try:
            from .services import get_detector
            get_detector()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Detector warmup skipped: {e}")
