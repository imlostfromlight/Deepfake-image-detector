import io

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings

from .services import combined_predict, detect_in_document, detect_video_url, robustness_attack_test
from .training import start_retraining, get_status as get_train_status
from .challenge import generate_challenge_token, verify_challenge_token, verify_frame_motion


class ChallengeView(APIView):
    """
    GET /api/live/challenge/
    Issues a signed one-time challenge token that the frontend must include
    when submitting live frames. Prevents direct API injection.
    """
    def get(self, request, *args, **kwargs):
        token = generate_challenge_token(settings.SECRET_KEY)
        return Response({"token": token, "ttl_seconds": 180})


class ImageDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        live = request.data.get('source') == 'live'

        if live:
            return self._handle_live(request)
        else:
            return self._handle_upload(request)

    # ── Live detection (webcam challenge) ─────────────────────────────────────

    def _handle_live(self, request):
        # 1. Validate challenge token — rejects any direct API call
        token = request.data.get('challenge_token', '')
        if not token:
            return Response(
                {"error": "No challenge token — live verification must go through the UI"},
                status=403,
            )
        valid, reason = verify_challenge_token(token, settings.SECRET_KEY)
        if not valid:
            return Response({"error": f"Challenge rejected: {reason}"}, status=403)

        # 2. Collect 3 frames (position, post-left-turn, post-right-turn)
        frames_bytes = []
        for i in range(3):
            f = request.data.get(f'frame_{i}')
            if not f:
                return Response({"error": f"Missing frame_{i}"}, status=400)
            frames_bytes.append(f.read())

        # 3. Server-side motion verification — rejects static images submitted 3×
        motion_ok, motion_reason = verify_frame_motion(frames_bytes)
        if not motion_ok:
            return Response({"error": f"Motion verification failed: {motion_reason}"}, status=400)

        # 4. Run deepfake + liveness detection on frame_2 (final frontal pose)
        result = combined_predict(io.BytesIO(frames_bytes[2]), live=True, explain=False)
        return Response(result)

    # ── Static image upload / video-call quick scan ───────────────────────────

    def _handle_upload(self, request):
        file = request.data.get('image')
        if not file:
            return Response({"error": "No image uploaded"}, status=400)
        source = request.data.get('source', 'upload')
        # videocall: skip heatmap/ELA generation for speed (3-second polling)
        explain = source != 'videocall'
        result = combined_predict(file, live=False, explain=explain)
        return Response(result)


class VideoLinkDetectionView(APIView):
    """POST /api/detect/video/ — deepfake analysis of a TikTok/Reel/Shorts URL."""

    def post(self, request, *args, **kwargs):
        url = (request.data.get('url') or '').strip()
        if not url:
            return Response({"error": "No URL provided"}, status=400)
        try:
            result = detect_video_url(url)
            return Response(result)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=400)
        except Exception:
            return Response(
                {"error": "Analysis failed — the video may be private, geo-restricted, or in an unsupported format"},
                status=500,
            )


class DocumentDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file = request.data.get('document')
        if not file:
            return Response({"error": "No document uploaded"}, status=400)

        result = detect_in_document(file.read(), file.name)
        if result.get("error") and not result.get("results"):
            return Response(result, status=400)
        return Response(result)



class FeedbackView(APIView):
    """
    POST /api/feedback/
    Stores a user-corrected label for an uploaded image.
    Fields: image (file), predicted ('real'|'deepfake'), true_label ('real'|'deepfake')
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        image      = request.data.get('image')
        true_label = (request.data.get('true_label') or '').strip()
        predicted  = (request.data.get('predicted')  or '').strip()

        if not image or true_label not in ('real', 'deepfake'):
            return Response({'error': 'image and true_label (real|deepfake) required'}, status=400)

        from .models import FeedbackSample
        FeedbackSample.objects.create(
            image=image,
            true_label=true_label,
            predicted=predicted,
            was_correct=(true_label == predicted),
        )
        pending = FeedbackSample.objects.filter(used_in_train=False).count()
        return Response({'status': 'saved', 'pending_samples': pending})


class RetrainView(APIView):
    """POST /api/retrain/ — launch background head fine-tuning."""

    def post(self, request, *args, **kwargs):
        started = start_retraining(settings.DEEPFAKE_MODEL_PATH)
        if not started:
            return Response({'error': 'Training already in progress'}, status=409)
        return Response({'status': 'started'})


class RetrainStatusView(APIView):
    """GET /api/retrain/status/ — training state + sample counts."""

    def get(self, request, *args, **kwargs):
        return Response(get_train_status())


class RobustnessTestView(APIView):
    """
    POST /api/detect/robustness/
    Takes the same image, applies 6 evasion attacks (JPEG compression, noise,
    blur, brightness shift, screenshot resize) and re-runs the ML ensemble on
    each variant.  Returns per-attack fake probabilities so the frontend can
    show that the detector is robust to common bypass techniques.
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        img_file = request.data.get('image')
        if not img_file:
            return Response({"error": "No image provided"}, status=400)
        try:
            from PIL import Image
            pil = Image.open(img_file).convert("RGB")
            result = robustness_attack_test(pil)
            return Response(result)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
