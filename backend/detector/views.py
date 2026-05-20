import io

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings

from .services import combined_predict, detect_in_document, detect_video_url
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

    # ── Static image upload ───────────────────────────────────────────────────

    def _handle_upload(self, request):
        file = request.data.get('image')
        if not file:
            return Response({"error": "No image uploaded"}, status=400)
        result = combined_predict(file, live=False, explain=True)
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
