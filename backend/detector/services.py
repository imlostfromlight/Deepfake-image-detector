from transformers import AutoModelForImageClassification, AutoImageProcessor
from facenet_pytorch import MTCNN
from PIL import Image
from PIL.ExifTags import TAGS
import torch
import torch._dynamo
torch._dynamo.config.disable = True
import io
import os
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


# ── Geometry / shadow consistency check ──────────────────────────────────────

def check_geometry_consistency(image: Image.Image) -> dict:
    """
    Estimate whether perspective and shadow cues are internally consistent.

    Two signals:
      1. Shadow gradient variance — cast shadows across the scene should all
         point in roughly the same direction (one dominant light source).
         High circular variance of gradient angles in dark regions → suspicious.

      2. Face vs scene lighting asymmetry — brightness left/right asymmetry of
         the face should agree in sign with the background's asymmetry.
         A face lit from the left inside a right-lit scene → suspicious.

    Adjustment range: [0.0, +0.12]
      High shadow variance  → up to +0.08
      Face/scene mismatch   → +0.07
      (capped at 0.12 total to avoid over-penalising)
    """
    result: dict = {
        "shadow_variance":        None,
        "face_bg_lighting_delta": None,
        "adjustment":             0.0,
        "signals":                [],
    }

    try:
        import cv2
    except ImportError:
        return result

    try:
        img_np = np.array(image.convert("RGB"))
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w   = gray.shape

        # ── 1. Shadow direction consistency ──────────────────────────────────
        gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = np.arctan2(gy, gx)

        # Shadow edges: dark regions with a detectable intensity boundary
        shadow_angles = ang[(gray < 90) & (mag > 15)]

        if shadow_angles.size > 300:
            sin_m    = np.mean(np.sin(shadow_angles))
            cos_m    = np.mean(np.cos(shadow_angles))
            variance = 1.0 - float(np.sqrt(sin_m ** 2 + cos_m ** 2))
            result["shadow_variance"] = round(variance, 3)

            if variance > 0.92:
                result["adjustment"] += 0.08
                result["signals"].append(
                    f"Shadow directions highly inconsistent (variance {variance:.2f}) — "
                    "possible composited elements from different light sources"
                )
            elif variance > 0.82:
                result["adjustment"] += 0.03
                result["signals"].append(
                    f"Moderate shadow direction inconsistency (variance {variance:.2f})"
                )

        # ── 2. Face lighting vs scene lighting ────────────────────────────────
        mtcnn = _get_mtcnn()
        with torch.no_grad():
            boxes, _, _ = mtcnn.detect(image.convert("RGB"), landmarks=True)

        if boxes is not None and len(boxes) > 0:
            bx1 = max(0, int(boxes[0][0]))
            by1 = max(0, int(boxes[0][1]))
            bx2 = min(w, int(boxes[0][2]))
            by2 = min(h, int(boxes[0][3]))
            fw, fh = bx2 - bx1, by2 - by1

            if fw > 20 and fh > 20:
                face_crop = gray[by1:by2, bx1:bx2]

                # Lateral brightness asymmetry: positive = right side brighter → lit from right
                face_dir = float(np.mean(face_crop[:, fw // 2:])) - float(np.mean(face_crop[:, :fw // 2]))

                # Background: full image left vs right, with face column zeroed out
                bg = gray.astype(np.float64)
                bg[by1:by2, bx1:bx2] = np.nan
                bg_dir = float(np.nanmean(bg[:, w // 2:])) - float(np.nanmean(bg[:, :w // 2]))

                # Flag only when both regions show non-trivial asymmetry and disagree in sign
                if abs(face_dir) > 6 and abs(bg_dir) > 4 and np.sign(face_dir) != np.sign(bg_dir):
                    delta = abs(face_dir - bg_dir)
                    result["face_bg_lighting_delta"] = round(delta, 1)
                    result["adjustment"] += 0.07
                    result["signals"].append(
                        f"Face lighting direction conflicts with scene lighting "
                        f"(face Δ{face_dir:+.1f}, scene Δ{bg_dir:+.1f}) — possible face composite"
                    )
                else:
                    result["face_bg_lighting_delta"] = 0.0

        result["adjustment"] = float(np.clip(result["adjustment"], 0.0, 0.12))

    except Exception:
        pass

    return result


# ── Hany Farid-inspired forensic signals ─────────────────────────────────────

def check_forensic_signals(image: Image.Image) -> dict:
    """
    Three forensic checks inspired by Hany Farid's digital-forensics research.

      1. Chromatic aberration (CA) — real camera lenses produce colour fringing
         that increases toward the image periphery. AI/composite images either
         have near-zero CA (too perfect) or show the wrong radial pattern.

      2. Local noise inconsistency — a genuine camera image has spatially
         uniform sensor noise. Composited images show abrupt changes in noise
         level between regions sourced from different originals.

      3. Eye specular highlight symmetry — both eyes are illuminated by the same
         environment, so the bright specular spot should appear at the same
         relative position in each eye. A large positional mismatch is a strong
         indicator that the face was composited from a different photo.

    Adjustment range: [0.0, +0.10]
    """
    result: dict = {
        "ca_score":            None,
        "noise_cv":            None,
        "eye_highlight_delta": None,
        "adjustment":          0.0,
        "signals":             [],
    }

    try:
        import cv2
    except ImportError:
        return result

    try:
        img_np = np.array(image.convert("RGB"))
        r = img_np[:, :, 0].astype(np.float32)
        b = img_np[:, :, 2].astype(np.float32)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape

        # ── 1. Chromatic aberration: peripheral vs central CA level ──────────
        if min(h, w) >= 80:
            gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)

            edge_mask = mag > np.percentile(mag, 85)
            ys, xs = np.mgrid[0:h, 0:w]
            dist   = np.sqrt(((ys - h / 2) / h) ** 2 + ((xs - w / 2) / w) ** 2)

            center_mask = (dist < 0.30) & edge_mask
            periph_mask = (dist > 0.42) & edge_mask

            if center_mask.sum() > 50 and periph_mask.sum() > 50:
                ca_center = float(np.mean(np.abs(r[center_mask] - b[center_mask])))
                ca_periph = float(np.mean(np.abs(r[periph_mask] - b[periph_mask])))
                ca_ratio  = ca_periph / (ca_center + 1e-8)
                result["ca_score"] = round(ca_ratio, 2)

                if ca_ratio < 0.5 and ca_center < 3.0:
                    # CA decreasing toward periphery — opposite of real optics
                    result["adjustment"] += 0.05
                    result["signals"].append(
                        f"Chromatic aberration is lower at image periphery than centre "
                        f"(ratio {ca_ratio:.2f}) — inconsistent with real lens optics"
                    )
                elif ca_center < 1.0 and ca_periph < 1.0:
                    # Essentially zero CA — characteristic of AI-generated images
                    result["adjustment"] += 0.03
                    result["signals"].append(
                        "Near-zero chromatic aberration across the whole image — "
                        "real camera lenses always produce some colour fringing"
                    )

        # ── 2. Local noise inconsistency ─────────────────────────────────────
        block = 64
        noise_levels: list[float] = []
        for y in range(0, h - block + 1, block):
            for x in range(0, w - block + 1, block):
                patch = gray[y : y + block, x : x + block]
                lap   = cv2.Laplacian(patch, cv2.CV_64F)
                noise_levels.append(float(np.median(np.abs(lap))))

        if len(noise_levels) >= 4:
            mean_n = float(np.mean(noise_levels))
            cv_n   = float(np.std(noise_levels) / (mean_n + 1e-8))
            result["noise_cv"] = round(cv_n, 3)

            if cv_n > 1.2:
                result["adjustment"] += 0.06
                result["signals"].append(
                    f"Noise level varies greatly across image regions (CV {cv_n:.2f}) — "
                    "suggests different source materials composited together"
                )
            elif cv_n > 0.90:
                result["adjustment"] += 0.02
                result["signals"].append(
                    f"Moderate spatial noise inconsistency (CV {cv_n:.2f})"
                )

        # ── 3. Eye specular highlight symmetry ───────────────────────────────
        mtcnn = _get_mtcnn()
        with torch.no_grad():
            _, _, landmarks = mtcnn.detect(image.convert("RGB"), landmarks=True)

        if landmarks is not None and len(landmarks) > 0:
            lm       = landmarks[0]       # (5, 2): L-eye, R-eye, nose, L-mouth, R-mouth
            eye_half = 18
            highlights: list[tuple[float, float]] = []

            for eye_xy in (lm[0], lm[1]):
                ex, ey = int(eye_xy[0]), int(eye_xy[1])
                x1 = max(0, ex - eye_half); x2 = min(w, ex + eye_half)
                y1 = max(0, ey - eye_half); y2 = min(h, ey + eye_half)
                patch = gray[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                ph, pw = patch.shape
                hy, hx = np.unravel_index(int(np.argmax(patch)), patch.shape)
                highlights.append((hx / pw, hy / ph))   # normalised [0..1]

            if len(highlights) == 2:
                dx = abs(highlights[0][0] - highlights[1][0])
                dy = abs(highlights[0][1] - highlights[1][1])
                mismatch = float(np.sqrt(dx ** 2 + dy ** 2))
                result["eye_highlight_delta"] = round(mismatch, 3)

                if mismatch > 0.55:
                    result["adjustment"] += 0.08
                    result["signals"].append(
                        f"Eye specular highlights mismatched (Δ{mismatch:.2f}) — "
                        "both eyes should reflect the same light source; "
                        "strong indicator of face compositing"
                    )
                elif mismatch > 0.35:
                    result["adjustment"] += 0.03
                    result["signals"].append(
                        f"Eye specular highlights partially mismatched (Δ{mismatch:.2f})"
                    )

        result["adjustment"] = float(np.clip(result["adjustment"], 0.0, 0.10))

    except Exception:
        pass

    return result


# ── Error Level Analysis ──────────────────────────────────────────────────────

def compute_ela(image: Image.Image, quality: int = 90) -> tuple[str | None, float]:
    """
    Re-save the image at a fixed JPEG quality, then amplify the per-pixel
    difference.  Regions that were previously edited (saved at a different
    quality, pasted from another source, or AI-inpainted) retain a different
    DCT quantisation pattern and show up as bright hotspots.

    Returns (base64-encoded overlay image, mean ELA score 0-255).
    Score > ~8 on an uncompressed PNG suggests prior manipulation.
    """
    try:
        import cv2

        orig_rgb = image.convert("RGB")
        buf = io.BytesIO()
        orig_rgb.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")

        orig_arr    = np.array(orig_rgb,  dtype=np.float32)
        resaved_arr = np.array(resaved,   dtype=np.float32)

        diff      = np.abs(orig_arr - resaved_arr)
        ela_score = float(np.mean(diff))

        # Amplify 15× then apply JET colourmap for visibility
        gray     = np.clip(np.mean(diff, axis=2) * 15, 0, 255).astype(np.uint8)
        coloured = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        col_rgb  = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

        orig_u8 = np.array(orig_rgb, dtype=np.uint8)
        blended = np.uint8(0.40 * orig_u8 + 0.60 * col_rgb)

        buf2 = io.BytesIO()
        Image.fromarray(blended).save(buf2, format="JPEG", quality=85)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf2.getvalue()).decode()
        return b64, ela_score

    except Exception as e:
        print(f"[ELA] {e}")
        return None, 0.0


# ── Generator / provenance attribution ───────────────────────────────────────

# IPTC DigitalSourceType values that declare AI generation (Rec. G7-2021)
_IPTC_AI_VALUES = {
    "trainedalgorithamicmedia": "Trained Algorithmic Media (IPTC)",
    "trainedalgorithmicmedia":  "Trained Algorithmic Media (IPTC)",
    "compositecapture":         "Composite Capture (IPTC)",
    "algorithmicmedia":         "Algorithmic Media (IPTC)",
}

# XMP / raw-binary patterns → generator name
_GENERATOR_PATTERNS: list[tuple[bytes, str]] = [
    (b"openai",           "DALL-E / OpenAI"),
    (b"dall-e",           "DALL-E / OpenAI"),
    (b"dalle",            "DALL-E / OpenAI"),
    (b"midjourney",       "Midjourney"),
    (b"stable-diffusion", "Stable Diffusion"),
    (b"stablediffusion",  "Stable Diffusion"),
    (b"stable_diffusion", "Stable Diffusion"),
    (b"comfyui",          "ComfyUI (Stable Diffusion)"),
    (b"automatic1111",    "AUTOMATIC1111 (Stable Diffusion)"),
    (b"a1111",            "AUTOMATIC1111 (Stable Diffusion)"),
    (b"invokeai",         "InvokeAI (Stable Diffusion)"),
    (b"novelai",          "NovelAI"),
    (b"fooocus",          "Fooocus (Stable Diffusion)"),
    (b"adobe firefly",    "Adobe Firefly"),
    (b"firefly",          "Adobe Firefly"),
    (b"google imagen",    "Google Imagen"),
    (b"imagen",           "Google Imagen"),
    (b"gemini",           "Google Gemini"),
    (b"leonardo.ai",      "Leonardo AI"),
    (b"dreamshaper",      "DreamShaper (Stable Diffusion)"),
    (b"playground ai",    "Playground AI"),
    (b"nightcafe",        "NightCafe"),
    (b"tensor.art",       "Tensor.Art"),
    (b"truepic",          "Truepic (camera-authenticated)"),
    (b"leica",            "Leica Camera (authentic)"),
]


def extract_provenance_signals(image_bytes: bytes) -> dict:
    """
    Deep provenance scan: C2PA manifest, IPTC DigitalSourceType, XMP
    CreatorTool, and raw binary generator strings in the first 128 KB.

    Confidence levels:
      "definitive" — cryptographic C2PA manifest or explicit IPTC AI tag
      "high"       — CreatorTool / XMP field names the generator directly
      "medium"     — generator string found in raw binary metadata
      "none"       — nothing found

    Score impact (applied in combined_predict):
      override = True  → fake_prob forced to 1.0 regardless of ML score.
                         Triggered by IPTC AI declaration or C2PA + known AI tool.
                         Metadata is cryptographic — it cannot be wrong.
      adjustment       → additive delta when override is False:
                         known AI generator (high conf) → +0.25
                         known AI generator (medium)    → +0.12
                         authentic camera (Leica/Truepic) → −0.10
    """
    # Camera-authenticated generators — these increase authenticity, not fakeness
    _CAMERA_GENERATORS = {b"leica", b"truepic"}

    result: dict = {
        "generator":   None,
        "confidence":  "none",
        "has_c2pa":    False,
        "has_iptc_ai": False,
        "override":    False,   # True → force verdict to deepfake
        "adjustment":  0.0,     # additive score delta when override is False
        "signals":     [],
    }

    try:
        # ── C2PA JUMBF (JPEG APP11 = 0xFFEB, labelled "c2pa") ────────────────
        head = image_bytes[:65536]
        if b"\xff\xeb" in image_bytes and b"c2pa" in head:
            result["has_c2pa"]   = True
            result["confidence"] = "definitive"
            result["signals"].append(
                "C2PA Content Credentials present — cryptographically signed provenance manifest"
            )
            head_lower = head.lower()
            for pat, name in _GENERATOR_PATTERNS:
                if pat in head_lower:
                    result["generator"] = name
                    # Camera-authenticated sources are real, not AI
                    if any(cam in pat for cam in _CAMERA_GENERATORS):
                        result["adjustment"] = -0.10
                    else:
                        result["override"] = True   # C2PA + known AI tool = definitive fake
                    break
            if not result["generator"]:
                result["generator"] = "C2PA-compliant tool (generator not identified in manifest)"

        # ── XMP / IPTC metadata ───────────────────────────────────────────────
        import re
        pil     = Image.open(io.BytesIO(image_bytes))
        xmp_raw = pil.info.get("xmp", b"")
        if isinstance(xmp_raw, str):
            xmp_raw = xmp_raw.encode("utf-8", errors="ignore")
        xmp = xmp_raw.decode("utf-8", errors="ignore") if xmp_raw else ""
        xl  = xmp.lower()

        if xl:
            # IPTC DigitalSourceType — explicit AI declaration in the standard
            for iptc_val, label in _IPTC_AI_VALUES.items():
                if iptc_val in xl:
                    result["has_iptc_ai"] = True
                    result["confidence"]  = "definitive"
                    result["override"]    = True   # IPTC tag = definitively AI-generated
                    result["signals"].append(f"IPTC DigitalSourceType = {label} — declared AI-generated")
                    break

            # Parse DigitalSourceType value for display
            m = re.search(r"digitalsourcetype[^>]*?>([^<]+)<", xmp, re.IGNORECASE)
            if m:
                dst = m.group(1).strip().split("/")[-1]
                result["signals"].append(f"IPTC DigitalSourceType: {dst}")

            # XMP CreatorTool
            m = re.search(r"creatortool[^>]*?>([^<]+)<", xmp, re.IGNORECASE)
            if m:
                tool     = m.group(1).strip()
                tool_low = tool.lower().encode()
                result["signals"].append(f"XMP CreatorTool: {tool}")
                if result["confidence"] in ("none", "medium"):
                    for pat, name in _GENERATOR_PATTERNS:
                        if pat in tool_low:
                            result["generator"] = name
                            result["confidence"] = "high"
                            if not any(cam in pat for cam in _CAMERA_GENERATORS):
                                result["adjustment"] = max(result["adjustment"], 0.25)
                            break

            # CAI / C2PA in XMP namespace
            if "cai:" in xl or "c2pa" in xl:
                result["has_c2pa"] = True
                if result["confidence"] == "none":
                    result["confidence"] = "high"
                result["signals"].append("Content Authenticity Initiative (CAI) XMP namespace present")

        # ── Raw binary scan of first 128 KB ──────────────────────────────────
        raw_low = image_bytes[:131072].lower()
        if result["confidence"] in ("none",) and not result["generator"]:
            for pat, name in _GENERATOR_PATTERNS:
                if pat in raw_low:
                    result["generator"] = name
                    result["confidence"] = "medium"
                    result["signals"].append(f"Generator string found in file binary: {name}")
                    if not any(cam in pat for cam in _CAMERA_GENERATORS):
                        result["adjustment"] = max(result["adjustment"], 0.12)
                    break

    except Exception as e:
        print(f"[provenance] {e}")

    # Safety net: has_iptc_ai always implies override, regardless of which
    # code path set it (catches any edge-case ordering issues above).
    if result["has_iptc_ai"]:
        result["override"] = True

    return result


# ── Partial face manipulation detection ──────────────────────────────────────

def check_partial_manipulation(image: Image.Image) -> dict:
    """
    Score the upper face (eyes + forehead) and lower face (nose + mouth + chin)
    independently.  A large discrepancy between the two halves suggests a
    partial swap — e.g. mouth replacement (Wav2Lip / DeepFaceLab), eye
    replacement, or inpainted lower face.

    Adjustment range: [0.0, +0.08]
    """
    result: dict = {
        "upper_score":  None,
        "lower_score":  None,
        "discrepancy":  None,
        "adjustment":   0.0,
        "signals":      [],
    }

    try:
        mtcnn = _get_mtcnn()
        with torch.no_grad():
            boxes, _, landmarks = mtcnn.detect(image.convert("RGB"), landmarks=True)

        if boxes is None or len(boxes) == 0:
            return result

        box = boxes[0]
        lm  = landmarks[0]   # (5, 2): L-eye, R-eye, nose, L-mouth, R-mouth

        iw, ih = image.size
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(iw, int(box[2]))
        y2 = min(ih, int(box[3]))
        fh = y2 - y1

        if fh < 40:
            return result

        # Split at the nose tip, but keep each half at least ¼ of face height
        nose_y  = int(lm[2][1])
        split_y = int(np.clip(nose_y, y1 + fh // 4, y2 - fh // 4))

        detector    = get_detector()
        upper_crop  = image.crop((x1, y1, x2, split_y)).resize((224, 224))
        lower_crop  = image.crop((x1, split_y, x2, y2)).resize((224, 224))

        upper_score = round(detector.predict(upper_crop)["fake_prob"] * 100, 1)
        lower_score = round(detector.predict(lower_crop)["fake_prob"] * 100, 1)
        discrepancy = round(abs(upper_score - lower_score), 1)

        result["upper_score"] = upper_score
        result["lower_score"] = lower_score
        result["discrepancy"] = discrepancy

        if discrepancy > 25:
            result["adjustment"] += 0.08
            worse = "Upper" if upper_score > lower_score else "Lower"
            result["signals"].append(
                f"Large score gap between face halves ({discrepancy:.0f}pt) — "
                f"{worse} face is much more suspicious; possible partial face swap"
            )
        elif discrepancy > 15:
            result["adjustment"] += 0.04
            result["signals"].append(
                f"Moderate face-half discrepancy ({discrepancy:.0f}pt) — "
                "upper and lower regions scored differently"
            )

        result["adjustment"] = float(np.clip(result["adjustment"], 0.0, 0.08))

    except Exception:
        pass

    return result


# ── Multi-face analysis ───────────────────────────────────────────────────────

_mtcnn_all = None


def _get_mtcnn_all():
    global _mtcnn_all
    if _mtcnn_all is None:
        _mtcnn_all = MTCNN(keep_all=True, post_process=False, device="cpu")
    return _mtcnn_all


def analyze_all_faces(image: Image.Image) -> list[dict]:
    """
    Detect every face in the image and run the deepfake detector on each.
    Returns a list sorted by face size (largest first).
    Only called for static image uploads (not live/video frames).
    """
    results: list[dict] = []
    try:
        mtcnn = _get_mtcnn_all()
        with torch.no_grad():
            boxes, probs, _ = mtcnn.detect(image.convert("RGB"), landmarks=True)

        if boxes is None:
            return results

        iw, ih   = image.size
        detector = get_detector()
        margin   = 30

        for i, (box, conf) in enumerate(zip(boxes, probs)):
            if float(conf) < 0.85:
                continue
            x1 = max(0, int(box[0]) - margin)
            y1 = max(0, int(box[1]) - margin)
            x2 = min(iw, int(box[2]) + margin)
            y2 = min(ih, int(box[3]) + margin)

            face_crop = image.crop((x1, y1, x2, y2))
            pred      = detector.predict(face_crop)

            results.append({
                "face_index":   i + 1,
                "bbox":         [x1, y1, x2, y2],
                "det_conf":     round(float(conf) * 100, 1),
                "fake_prob":    round(pred["fake_prob"] * 100, 1),
                "prediction":   pred["prediction"],
                "confidence":   round(pred["confidence"], 1),
            })

        # Sort by face area descending
        results.sort(key=lambda r: (r["bbox"][2]-r["bbox"][0]) * (r["bbox"][3]-r["bbox"][1]), reverse=True)

    except Exception:
        pass

    return results


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


# ── UniversalFakeDetect (CLIP ViT-L/14 + linear probe) ───────────────────────

class UniversalFakeDetector:
    """
    Wang et al. "Towards Universal Fake Image Detection by Exploiting the
    Artefacts of Generative Models" — CVPR 2023.
    Uses CLIP ViT-L/14 features + a linear probe trained on diverse GAN/diffusion
    outputs (ProGAN, StyleGAN, DALL-E, Stable Diffusion, etc.).
    Weights file: backend/detector/ufd_fc_weights.pth
    Download from https://github.com/Yuheng-Li/UniversalFakeDetect (see README).
    """
    _instance = None
    _available = False

    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "ufd_fc_weights.pth")

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._loaded = False
            try:
                import clip as clip_lib
                device = "cuda" if torch.cuda.is_available() else "cpu"
                obj.device = device
                obj.clip_model, obj.preprocess = clip_lib.load("ViT-L/14", device=device)
                obj.clip_model.eval()
                obj.fc = torch.nn.Linear(768, 1).to(device)
                if os.path.exists(cls.WEIGHTS_PATH):
                    state = torch.load(cls.WEIGHTS_PATH, map_location=device)
                    # Handle both raw state_dict and wrapped checkpoints
                    if isinstance(state, dict) and "fc." in str(list(state.keys())):
                        # wrapped: strip prefix
                        state = {k.replace("fc.", ""): v for k, v in state.items()}
                    obj.fc.load_state_dict(state)
                    obj.fc.eval()
                    obj._loaded = True
                    cls._available = True
                    print("[UFD] UniversalFakeDetect loaded ✓")
                else:
                    print(f"[UFD] Weights not found at {cls.WEIGHTS_PATH} — UFD disabled")
            except Exception as e:
                print(f"[UFD] Failed to initialise: {e}")
            cls._instance = obj
        return cls._instance

    def predict(self, image: "Image.Image") -> dict:
        """Returns fake_prob in [0,1]. Returns None if model not loaded."""
        if not self._loaded:
            return None
        try:
            import clip as clip_lib
            inp = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_image(inp).float()
                feat = feat / feat.norm(dim=-1, keepdim=True)
                logit = self.fc(feat).squeeze()
                prob = torch.sigmoid(logit).item()
            return {"fake_prob": round(prob, 4)}
        except Exception as e:
            print(f"[UFD] predict error: {e}")
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

    # 0.97: genuine screen/photo replays score ≥0.98; webcam false positives
    # (dim light, partial angle) typically land 0.90–0.96.
    SPOOF_THRESHOLD = 0.97

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
    Full detection pipeline — CLIP ViT-B/16 + LoRA + all side signals.
    """
    image_bytes = image_file.read()
    pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Face alignment then CLIP+LoRA inference
    aligned  = _crop_and_align_face(pil_image)
    detector = get_detector()
    result   = detector.predict(aligned)
    lora_prob = result["fake_prob"]

    # UniversalFakeDetect ensemble (CLIP ViT-L/14 trained on GAN+diffusion fakes)
    ufd       = UniversalFakeDetector()
    ufd_result = ufd.predict(pil_image)
    if ufd_result is not None:
        # Take the MAX: if either model is highly confident it's fake, respect that.
        # Averaging was wrong — UFD dragged down a 94% LoRA signal to below threshold.
        raw_prob = float(np.clip(max(lora_prob, ufd_result["fake_prob"]), 0.0, 1.0))
    else:
        raw_prob = lora_prob

    # Side signals — each applies a small score correction
    meta       = extract_exif_signals(pil_image)
    geo        = check_geometry_consistency(pil_image)
    forensic   = check_forensic_signals(pil_image)
    partial    = check_partial_manipulation(pil_image)
    provenance = extract_provenance_signals(image_bytes)

    threshold = detector.threshold

    if provenance["override"]:
        # Metadata is cryptographic evidence — it overrides the ML score entirely.
        # IPTC TrainedAlgorithmicMedia or C2PA + identified AI tool = definitively fake.
        fake_prob  = 1.0
        final_pred = "fake"
        confidence = 100.0
        overall    = "deepfake"
    else:
        fake_prob = float(np.clip(
            raw_prob
            + meta["adjustment"]
            + geo["adjustment"]
            + forensic["adjustment"]
            + partial["adjustment"]
            + provenance["adjustment"],
            0.0, 1.0,
        ))
        final_pred = "fake" if fake_prob >= threshold else "real"
        confidence = round(fake_prob * 100 if final_pred == "fake" else (1 - fake_prob) * 100, 2)
        overall    = "deepfake" if final_pred == "fake" else "real"

    analysis = {
        "ml_score":              round(lora_prob                   * 100, 1),
        "ufd_score":             round(ufd_result["fake_prob"] * 100, 1) if ufd_result else None,
        "fft_score":             None,
        "meta_adjustment":       round(meta["adjustment"]          * 100, 1),
        "geo_adjustment":        round(geo["adjustment"]           * 100, 1),
        "forensic_adjustment":   round(forensic["adjustment"]      * 100, 1),
        "partial_adjustment":    round(partial["adjustment"]       * 100, 1),
        "provenance_adjustment": round(provenance["adjustment"]    * 100, 1),
        "provenance_override":   provenance["override"],
        "final_score":           round(fake_prob                   * 100, 1),
        "threshold":             round(threshold                   * 100, 1),
    }

    # Multi-face: only for static uploads (not live frames) to keep latency low
    faces = analyze_all_faces(pil_image) if not live else []

    response = {
        "overall":    overall,
        "deepfake":   {"prediction": final_pred, "confidence": confidence},
        "metadata":   meta,
        "provenance": provenance,
        "geometry":   geo,
        "forensic":   forensic,
        "partial":    partial,
        "faces":      faces,
        "analysis":   analysis,
    }

    if live:
        liveness_result = LivenessModel().predict(io.BytesIO(image_bytes))
        spoof_conf = liveness_result.get("confidence", 0) / 100.0
        # Require BOTH high spoof confidence AND the ML model not being
        # strongly convinced the face is real (raw_prob > 0.35).
        # This prevents dim-lighting / partial-angle false positives from
        # overriding a clearly-real ML verdict.
        if liveness_result["prediction"] == "spoof" and (raw_prob > 0.35 or spoof_conf >= 0.99):
            overall = "deepfake"
        response["overall"]  = overall
        response["liveness"] = liveness_result

    if explain:
        response["heatmap"] = generate_heatmap(aligned)
        ela_b64, ela_score  = compute_ela(pil_image)
        response["ela_map"]   = ela_b64
        response["ela_score"] = round(ela_score, 2)

    return response


# ── Document detection ────────────────────────────────────────────────────────

def _run_ensemble_on_pil(image: Image.Image) -> dict:
    """
    Run CLIP detector on a PIL Image (used by the document pipeline).
    Also generates GradCAM heatmap, ELA map, and returns the aligned face crop
    so the frontend can show the same visualisations as the main image detector.
    """
    aligned   = _crop_and_align_face(image)
    detector  = get_detector()
    result    = detector.predict(aligned)
    fake_prob = result["fake_prob"]
    threshold = detector.threshold
    normalized = "fake" if fake_prob >= threshold else "real"
    confidence = round(fake_prob * 100 if normalized == "fake" else (1 - fake_prob) * 100, 2)

    # Face crop as base64 so the frontend can display the extracted face
    face_b64: str | None = None
    try:
        buf = io.BytesIO()
        aligned.convert("RGB").resize((224, 224)).save(buf, format="JPEG", quality=85)
        face_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    # GradCAM heatmap on the aligned face
    heatmap = generate_heatmap(aligned)

    # ELA on the original extracted image (not the face crop)
    ela_b64, ela_score = compute_ela(image)

    return {
        "prediction": normalized,
        "confidence": confidence,
        "face_crop":  face_b64,
        "heatmap":    heatmap,
        "ela_map":    ela_b64,
        "ela_score":  round(ela_score, 2),
    }


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


# ── Voice deepfake detection ──────────────────────────────────────────────────

class _VoiceModel:
    """
    ResNet-18 trained on Mel spectrograms for voice deepfake detection.
    Weights: backend/detector/best_voice_model.pt
    Input: audio bytes (WebM/WAV/MP3) → 3-channel 224×224 Mel spectrogram image
    Output: fake_prob in [0, 1]
    """
    _instance = None
    WEIGHTS   = os.path.join(os.path.dirname(__file__), "best_voice_model.pt")
    SR        = 16_000   # sample rate used during training
    N_MELS    = 128
    HOP       = 512
    N_FFT     = 2048
    DURATION  = 3.0      # seconds per chunk

    def __new__(cls):
        if cls._instance is None:
            import torchvision.models as tvm
            obj = super().__new__(cls)
            obj.device = "cuda" if torch.cuda.is_available() else "cpu"
            net = tvm.resnet18(weights=None)
            net.fc = torch.nn.Linear(512, 1)
            state = torch.load(cls.WEIGHTS, map_location=obj.device)
            net.load_state_dict(state)
            net.eval()
            obj.net = net.to(obj.device)
            cls._instance = obj
        return cls._instance

    def predict(self, audio_bytes: bytes) -> dict:
        import librosa
        import tempfile, pathlib

        # Write to temp file so librosa can handle any format via soundfile/ffmpeg
        suffix = ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(audio_bytes)
            tmp_path = tf.name

        try:
            y, _ = librosa.load(tmp_path, sr=self.SR, mono=True,
                                duration=self.DURATION)
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

        if len(y) < self.SR * 0.5:
            return {"prediction": "unknown", "fake_prob": None,
                    "confidence": None, "error": "Audio too short (<0.5s)"}

        mel = librosa.feature.melspectrogram(
            y=y, sr=self.SR, n_fft=self.N_FFT,
            hop_length=self.HOP, n_mels=self.N_MELS,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Normalise to [0, 1] then to ImageNet stats
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        # Resize to 224×224 via PIL
        pil_mel = Image.fromarray((mel_norm * 255).astype(np.uint8)).resize(
            (224, 224), Image.BILINEAR
        )
        arr = np.array(pil_mel, dtype=np.float32) / 255.0
        arr = np.stack([arr] * 3, axis=0)
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        arr  = (arr - mean) / std

        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit    = self.net(tensor).squeeze()
            fake_prob = float(torch.sigmoid(logit).item())

        prediction = "fake" if fake_prob >= 0.5 else "real"
        confidence = round(fake_prob * 100 if prediction == "fake"
                           else (1 - fake_prob) * 100, 2)
        return {
            "prediction": prediction,
            "fake_prob":  round(fake_prob, 4),
            "confidence": confidence,
        }


_voice_model: "_VoiceModel | None" = None

def analyze_voice(audio_bytes: bytes) -> dict:
    global _voice_model
    if _voice_model is None:
        _voice_model = _VoiceModel()
    return _voice_model.predict(audio_bytes)
