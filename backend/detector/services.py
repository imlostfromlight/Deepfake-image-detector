from transformers import AutoModelForImageClassification, AutoImageProcessor
from facenet_pytorch import MTCNN
from PIL import Image
from PIL.ExifTags import TAGS
import torch
import torch._dynamo
torch._dynamo.config.disable = True
import io
import base64

import fitz   # PyMuPDF
import numpy as np

from .clip_detector import DeepfakeDetector, _PREPROCESS


# ── Face detection + alignment ───────────────────────────────────────────────

_mtcnn = None

def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=False, post_process=False, device="cpu")
    return _mtcnn


@torch._dynamo.disable
def _crop_and_align_face(image: Image.Image, margin: int = 40) -> Image.Image:
    """
    Detect the largest face, rotate so the eye line is horizontal, then crop.

    Why this matters: CLIP ViT-B/16 (like all ViT models) was trained on roughly
    frontal, aligned faces. A tilted or off-centre face causes inconsistent
    predictions. MTCNN returns 5 landmarks (L-eye, R-eye, nose, L-mouth,
    R-mouth) that let us compute the exact rotation to correct pose before
    cropping.
    """
    try:
        mtcnn = _get_mtcnn()
        with torch.no_grad():
            boxes, _, landmarks = mtcnn.detect(image, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return image

        box = boxes[0]
        lm  = landmarks[0]   # (5, 2)

        left_eye  = lm[0]
        right_eye = lm[1]
        dy    = float(right_eye[1] - left_eye[1])
        dx    = float(right_eye[0] - left_eye[0])
        angle = np.degrees(np.arctan2(dy, dx))

        eye_cx = float((left_eye[0] + right_eye[0]) / 2)
        eye_cy = float((left_eye[1] + right_eye[1]) / 2)
        aligned = image.rotate(-angle, center=(eye_cx, eye_cy), resample=Image.BILINEAR)

        w, h = aligned.size
        x1 = max(0, int(box[0]) - margin)
        y1 = max(0, int(box[1]) - margin)
        x2 = min(w, int(box[2]) + margin)
        y2 = min(h, int(box[3]) + margin)
        return aligned.crop((x1, y1, x2, y2))

    except Exception:
        return image


# ── EXIF metadata analysis ────────────────────────────────────────────────────

# Common screen resolutions — exact match suggests screenshot rather than camera photo
_SCREEN_RESOLUTIONS = {
    (1280, 720), (1366, 768), (1440, 900), (1536, 864),
    (1600, 900), (1920, 1080), (2048, 1152), (2560, 1080),
    (2560, 1440), (2560, 1600), (3440, 1440), (3840, 1080),
    (3840, 2160), (5120, 1440), (5120, 2880),
    # Common phone screen resolutions
    (390, 844), (393, 852), (414, 896), (428, 926),
    (1080, 1920), (1080, 2340), (1080, 2400), (1170, 2532),
    (1179, 2556), (1284, 2778), (1290, 2796),
}

# Software strings that definitively identify AI-generated images.
# Matched case-insensitively against the EXIF Software and ImageDescription fields.
_AI_GENERATOR_KEYWORDS = [
    "stable diffusion", "dall-e", "dall·e", "midjourney", "adobe firefly",
    "comfyui", "automatic1111", "a1111", "invokeai", "dreamstudio",
    "fooocus", "novelai", "diffusers", "leonardo ai", "playground ai",
    "nightcafe", "dreamshaper", "tensor.art", "getimg.ai",
]

# Software that edits real photos but doesn't generate them (suspicious, not conclusive).
_EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "capture one", "snapseed", "darktable",
]


def _check_screenshot_dimensions(image: Image.Image) -> str | None:
    """Return a label if the image dimensions exactly match a known screen resolution."""
    w, h = image.size
    if (w, h) in _SCREEN_RESOLUTIONS or (h, w) in _SCREEN_RESOLUTIONS:
        return f"{w}×{h} matches a known screen resolution"
    return None


def extract_exif_signals(image: Image.Image) -> dict:
    """
    Analyse EXIF metadata for authenticity and AI-generation signals.

    Adjustment range [-0.30, +0.30]:
      Strong real-camera evidence  → up to -0.30 (camera + GPS + optics + timestamp)
      Known AI generator software  → +0.25  (explicit tag in EXIF Software field)
      Photo editing software       → +0.05  (suspicious but not conclusive)
      No EXIF                      → neutral (0.0) — social media strips it from real photos too
    """
    result = {
        "has_exif":             False,
        "camera":               None,
        "software":             None,
        "has_gps":              False,
        "captured_at":          None,
        "image_source":         "unknown",
        "authenticity_signals": [],
        "suspicion_signals":    [],
        "adjustment":           0.0,
    }

    screen_hint = _check_screenshot_dimensions(image)

    try:
        exif_data = image.getexif()
        if not exif_data:
            if screen_hint:
                result["image_source"] = "screenshot"
                result["suspicion_signals"].append(f"Likely screenshot — {screen_hint}")
                result["suspicion_signals"].append("No EXIF — ML model score is the primary signal")
                result["adjustment"] = 0.0
            else:
                result["image_source"] = "web"
                result["suspicion_signals"].append(
                    "No EXIF metadata — social media strips it from real photos too, "
                    "so absence alone is not evidence of fakery."
                )
                result["adjustment"] = 0.0
            return result

        result["has_exif"] = True
        tagged = {TAGS.get(k, k): v for k, v in exif_data.items()}
        adj = 0.0

        # ── Check for explicit AI generator tag first ─────────────────────────
        software = str(tagged.get("Software", "")).strip()
        description = str(tagged.get("ImageDescription", "")).strip()
        comment = str(tagged.get("UserComment", b"")).strip()
        combined_text = f"{software} {description} {comment}".lower()

        result["software"] = software or None

        ai_match = next(
            (kw for kw in _AI_GENERATOR_KEYWORDS if kw in combined_text), None
        )
        if ai_match:
            adj += 0.25
            result["image_source"] = "ai_generated"
            result["suspicion_signals"].append(
                f"AI generator identified in metadata: '{software or description or comment}'"
            )
        elif any(kw in combined_text for kw in _EDITING_SOFTWARE_KEYWORDS):
            adj += 0.05
            result["suspicion_signals"].append(f"Post-processed with {software}")

        # ── Real-camera authenticity signals ──────────────────────────────────

        # Camera make + model: strongest single signal — AI images never have this
        make  = str(tagged.get("Make",  "")).strip()
        model = str(tagged.get("Model", "")).strip()
        if make or model:
            result["camera"] = f"{make} {model}".strip()
            adj -= 0.15
            result["authenticity_signals"].append(f"Camera hardware: {result['camera']}")

        # GPS — AI images are never taken anywhere
        if 34853 in exif_data:
            result["has_gps"] = True
            adj -= 0.10
            result["authenticity_signals"].append("GPS coordinates embedded")

        # Optical parameters (aperture, shutter speed, ISO)
        if "FNumber" in tagged or "ExposureTime" in tagged or "ISOSpeedRatings" in tagged:
            adj -= 0.08
            result["authenticity_signals"].append("Optical parameters present (aperture / shutter / ISO)")

        # Capture timestamp from the original shot
        dt = tagged.get("DateTimeOriginal") or tagged.get("DateTime")
        if dt:
            result["captured_at"] = str(dt)
            adj -= 0.05
            result["authenticity_signals"].append(f"Capture timestamp: {dt}")

        result["adjustment"] = float(np.clip(adj, -0.30, 0.30))

    except Exception:
        pass   # non-JPEG or unreadable EXIF — leave defaults

    return result


# ── CLIP detector singleton ───────────────────────────────────────────────────

_detector = None


def get_detector() -> DeepfakeDetector:
    global _detector
    if _detector is None:
        from django.conf import settings
        _detector = DeepfakeDetector(
            checkpoint_path=settings.DEEPFAKE_MODEL_PATH,
            threshold=settings.DEEPFAKE_THRESHOLD,
        )
    return _detector


# ── Attention Rollout heatmap (CLIP ViT-B/16) ────────────────────────────────

def generate_heatmap(face_image: Image.Image) -> str | None:
    """
    Attention Rollout heatmap using the fine-tuned CLIP ViT-B/16 backbone.

    CLIP ViT-B/16 processes the image as 14×14 = 196 patches + 1 CLS token,
    identical layout to the old PrithivModel ViT. We multiply attention matrices
    through all 12 encoder layers (Abnar & Zuidema 2020) to propagate how each
    patch's information reaches the CLS token. Red = high influence on verdict.

    No backward pass or hooks required — works directly on the loaded model.
    """
    try:
        import cv2
    except ImportError:
        return None

    try:
        detector  = get_detector()
        img_size  = 224
        n_patches = 14

        face_rgb = face_image.convert('RGB')
        x = _PREPROCESS(face_rgb).unsqueeze(0).to(detector.device)

        bb = detector.model.bb   # LoRA-wrapped CLIPVisionTransformer

        # Force eager attention so attention weights are actually stored.
        # SDPA / Flash Attention skip this for performance but break rollout.
        try:
            inner = bb.base_model.model if hasattr(bb, 'base_model') else bb
            inner.config._attn_implementation = 'eager'
        except Exception:
            pass

        with torch.no_grad():
            outputs = bb(pixel_values=x, output_attentions=True)

        if not outputs.attentions:
            return None

        seq_len = n_patches * n_patches + 1   # 197
        rollout = torch.eye(seq_len)

        for layer_attn in outputs.attentions:
            avg = layer_attn.squeeze(0).mean(dim=0).cpu()
            aug = avg + torch.eye(seq_len)
            aug = aug / aug.sum(dim=-1, keepdim=True)
            rollout = aug @ rollout

        # Row 0 = CLS token; columns 1..196 = patch positions
        patch_importance = rollout[0, 1:].numpy()
        heatmap_2d = patch_importance.reshape(n_patches, n_patches)

        lo, hi = heatmap_2d.min(), heatmap_2d.max()
        heatmap_2d = (heatmap_2d - lo) / (hi - lo + 1e-8)

        face_224    = face_rgb.resize((img_size, img_size))
        mask        = cv2.resize(heatmap_2d, (img_size, img_size))
        coloured    = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        coloured_rgb = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

        original_np = np.array(face_224, dtype=np.uint8)
        blended     = np.uint8(0.55 * original_np + 0.45 * coloured_rgb)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        print(f"[heatmap] Attention rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ── Anti-spoofing / liveness model ───────────────────────────────────────────

class LivenessModel:
    """
    Anti-spoofing: nguyenkhoa/mobilevitv2_Liveness_detection_v1.0.
    MobileViT-V2, 99.88% accuracy. Catches video-replay / phone-screen attacks.
    Threshold 0.80 (raised from 0.75) to reduce false positives on webcam.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            n = "nguyenkhoa/mobilevitv2_Liveness_detection_v1.0"
            obj.processor = AutoImageProcessor.from_pretrained(n)
            obj.model     = AutoModelForImageClassification.from_pretrained(n)
            cls._instance = obj
        return cls._instance

    # Raised from 0.80 → 0.92: genuine video-replay spoofs score >0.92;
    # lower values catch motion-blurred webcam frames as false positives.
    SPOOF_THRESHOLD = 0.92

    def predict(self, image_file):
        image = Image.open(image_file).convert("RGB")
        image = _crop_and_align_face(image)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(
                self.model(**inputs).logits, dim=1
            ).squeeze(0)
        live_prob  = float(probs[0])
        spoof_prob = float(probs[1])
        normalized = "spoof" if spoof_prob >= self.SPOOF_THRESHOLD else "live"
        confidence = round(spoof_prob * 100 if normalized == "spoof" else live_prob * 100, 2)
        return {"prediction": normalized, "confidence": confidence}


# ── Public API ────────────────────────────────────────────────────────────────

def combined_predict(image_file, live=False, explain=False):
    """
    Full detection pipeline — CLIP ViT-B/16 + LoRA + EXIF correction.
    Returns the same JSON shape as before so the frontend needs no changes.
    """
    image_bytes = image_file.read()
    pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Face alignment then CLIP inference
    aligned  = _crop_and_align_face(pil_image)
    detector = get_detector()
    result   = detector.predict(aligned)
    raw_prob = result['fake_prob']

    # EXIF metadata: kept as a side signal, applies a small score correction
    meta      = extract_exif_signals(pil_image)
    fake_prob = float(np.clip(raw_prob + meta["adjustment"], 0.0, 1.0))

    threshold  = detector.threshold
    final_pred = "fake" if fake_prob >= threshold else "real"
    confidence = round(fake_prob * 100 if final_pred == "fake" else (1 - fake_prob) * 100, 2)

    overall = "deepfake" if final_pred == "fake" else "real"

    # ml_score = raw CLIP output; fft_score removed (no FFT in new pipeline)
    analysis = {
        "ml_score":        round(raw_prob          * 100, 1),
        "fft_score":       None,
        "meta_adjustment": round(meta["adjustment"] * 100, 1),
        "final_score":     round(fake_prob          * 100, 1),
        "threshold":       round(threshold          * 100, 1),
    }

    response = {
        "overall":  overall,
        "deepfake": {"prediction": final_pred, "confidence": confidence},
        "metadata": meta,
        "analysis": analysis,
    }

    if live:
        liveness_result = LivenessModel().predict(io.BytesIO(image_bytes))
        if liveness_result["prediction"] == "spoof":
            overall = "deepfake"
        response["overall"]  = overall
        response["liveness"] = liveness_result

    if explain:
        response["heatmap"] = generate_heatmap(aligned)

    return response


# ── Document detection ────────────────────────────────────────────────────────

def _run_ensemble_on_pil(image: Image.Image) -> dict:
    """Run CLIP detector on a PIL Image (used by the document pipeline)."""
    aligned   = _crop_and_align_face(image)
    detector  = get_detector()
    result    = detector.predict(aligned)
    fake_prob = result['fake_prob']
    threshold = detector.threshold
    normalized = "fake" if fake_prob >= threshold else "real"
    confidence = round(fake_prob * 100 if normalized == "fake" else (1 - fake_prob) * 100, 2)
    return {"prediction": normalized, "confidence": confidence}


def _extract_images_from_pdf(pdf_bytes: bytes) -> list:
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                pil_image  = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                images.append({"page": page_num + 1, "index": img_idx + 1, "image": pil_image})
            except Exception:
                continue
    return images


def detect_in_document(file_bytes: bytes, filename: str) -> dict:
    """Analyse all face images embedded in a document (PDF or standalone image)."""
    ext     = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    results = []

    if ext in ("jpg", "jpeg", "png", "webp", "bmp", "gif"):
        try:
            pil_image       = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            deepfake_result = _run_ensemble_on_pil(pil_image)
            results.append({
                "page": 1, "index": 1,
                "overall": "deepfake" if deepfake_result["prediction"] == "fake" else "real",
                "deepfake": deepfake_result,
            })
        except Exception as e:
            return {"error": f"Failed to process image: {e}", "results": []}

    elif ext == "pdf":
        try:
            pdf_images = _extract_images_from_pdf(file_bytes)
        except Exception as e:
            return {"error": f"Failed to open PDF: {e}", "results": []}
        if not pdf_images:
            return {"error": "No images found in PDF", "results": []}
        for item in pdf_images:
            try:
                deepfake_result = _run_ensemble_on_pil(item["image"])
                results.append({
                    "page": item["page"], "index": item["index"],
                    "overall": "deepfake" if deepfake_result["prediction"] == "fake" else "real",
                    "deepfake": deepfake_result,
                })
            except Exception:
                continue
    else:
        return {"error": f"Unsupported file type: .{ext}", "results": []}

    deepfakes_found = sum(1 for r in results if r["overall"] == "deepfake")
    return {
        "filename":        filename,
        "total_analyzed":  len(results),
        "deepfakes_found": deepfakes_found,
        "results":         results,
    }


# ── Social media video detection ──────────────────────────────────────────────

# SSRF guard: only these domains are allowed as video sources
_ALLOWED_VIDEO_DOMAINS = frozenset({
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com", "m.tiktok.com",
    "instagram.com", "www.instagram.com",
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
})

MAX_VIDEO_FRAMES   = 30     # max frames to analyse per video
MAX_VIDEO_DURATION = 300    # seconds — reject videos longer than 5 min


def _validate_video_url(url: str) -> tuple[bool, str]:
    from urllib.parse import urlparse
    try:
        p = urlparse(url.strip())
        if p.scheme not in ("http", "https"):
            return False, "URL must start with http:// or https://"
        if p.netloc.lower() not in _ALLOWED_VIDEO_DOMAINS:
            return False, (
                "Unsupported platform. Paste a TikTok, Instagram Reel, "
                "or YouTube Shorts link."
            )
        return True, ""
    except Exception:
        return False, "Invalid URL format"


def _platform_name(extractor: str) -> str:
    e = extractor.lower()
    if "tiktok"    in e: return "TikTok"
    if "instagram" in e: return "Instagram"
    if "youtube"   in e: return "YouTube"
    return extractor.capitalize() or "Unknown"


def detect_video_url(url: str) -> dict:
    """
    Download a short-form social video, extract up to MAX_VIDEO_FRAMES face
    frames, run the CLIP detector on each, and return per-frame results plus
    an aggregated verdict.

    Supported: TikTok, Instagram Reels, YouTube Shorts.
    """
    import yt_dlp
    import cv2
    import tempfile
    import glob
    import os

    valid, err = _validate_video_url(url)
    if not valid:
        raise ValueError(err)

    meta: dict = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_tmpl = os.path.join(tmpdir, "video.%(ext)s")

        ydl_opts = {
            "outtmpl":       output_tmpl,
            # Prefer smallest MP4; fall back to smallest available stream
            "format":        "worstvideo[ext=mp4]/worstvideo/worst[ext=mp4]/worst",
            "quiet":         True,
            "no_warnings":   True,
            "max_filesize":  80 * 1024 * 1024,   # 80 MB hard cap
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True) or {}
                meta = {
                    "platform": _platform_name(info.get("extractor", "")),
                    "title":    info.get("title", "Unknown"),
                    "duration": info.get("duration") or 0,
                }
        except yt_dlp.utils.DownloadError as exc:
            msg = str(exc)
            if "login" in msg.lower() or "private" in msg.lower():
                raise ValueError("Video is private or requires login — only public videos are supported")
            raise ValueError(f"Download failed: {msg}")
        except Exception as exc:
            raise ValueError(f"Could not download video: {exc}")

        if meta["duration"] > MAX_VIDEO_DURATION:
            raise ValueError(
                f"Video is {meta['duration']}s — maximum allowed is {MAX_VIDEO_DURATION}s"
            )

        files = glob.glob(os.path.join(tmpdir, "video.*"))
        if not files:
            raise ValueError("Downloader produced no output file")
        video_path = files[0]

        # ── Frame extraction ───────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        sample_step  = max(1, total_frames // MAX_VIDEO_FRAMES)

        detector     = get_detector()
        frame_results: list[dict] = []

        for frame_idx in range(0, total_frames, sample_step):
            if len(frame_results) >= MAX_VIDEO_FRAMES:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = cap.read()
            if not ret:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            # Skip frames where no face is detected
            aligned = _crop_and_align_face(pil_img)
            if aligned is pil_img:     # _crop_and_align_face returns the input unchanged when no face found
                continue

            pred = detector.predict(aligned)
            frame_results.append({
                "timestamp":  round(frame_idx / fps, 1),
                "prediction": pred["prediction"],
                "fake_prob":  round(pred["fake_prob"] * 100, 1),
                "confidence": round(pred["confidence"], 1),
            })

        cap.release()

    if not frame_results:
        raise ValueError("No faces detected in the video — try a video where a face is clearly visible")

    # ── Aggregate verdict ──────────────────────────────────────────────────
    probs      = [r["fake_prob"] / 100 for r in frame_results]
    avg_prob   = sum(probs) / len(probs)
    max_prob   = max(probs)
    fake_count = sum(1 for r in frame_results if r["prediction"] == "fake")
    threshold  = detector.threshold

    overall_fake = avg_prob >= threshold

    return {
        **meta,
        "overall":  "deepfake" if overall_fake else "real",
        "summary": {
            "avg_fake_prob": round(avg_prob * 100, 1),
            "max_fake_prob": round(max_prob * 100, 1),
            "fake_frames":   fake_count,
            "total_frames":  len(frame_results),
            "threshold":     round(threshold * 100, 1),
        },
        "frames": frame_results,
    }
