"""
Server-side challenge-response for live deepfake detection.

Defenses provided:
  1. HMAC-signed token — cannot be forged without SECRET_KEY
  2. Short TTL (3 min) — replay window is tiny
  3. One-time nonce — each token can only be used once
  4. Multi-frame motion proof — 3 frames must show genuine head movement
"""
import hmac
import hashlib
import json
import base64
import time
import secrets
import threading

import numpy as np
from PIL import Image
from io import BytesIO

TOKEN_TTL_SECONDS = 180           # 3 minutes
MOTION_THRESHOLD  = 12.0         # mean absolute pixel diff (0-255 scale) per channel
IDENTICAL_THRESHOLD = 1.5        # diffs below this = same image submitted twice


# ── One-time nonce store ──────────────────────────────────────────────────────

_used_nonces: dict[str, float] = {}   # nonce → expiry unix timestamp
_lock = threading.Lock()


def _purge_expired() -> None:
    now = time.time()
    stale = [k for k, exp in _used_nonces.items() if exp < now]
    for k in stale:
        del _used_nonces[k]


# ── Token generation ──────────────────────────────────────────────────────────

def generate_challenge_token(secret_key: str) -> str:
    """
    Build a signed token: base64(json_payload) + "." + HMAC-SHA256.
    The payload carries a random nonce and an issue timestamp.
    """
    payload = base64.urlsafe_b64encode(
        json.dumps({"nonce": secrets.token_hex(16), "ts": int(time.time())}).encode()
    ).decode()
    sig = hmac.new(secret_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


# ── Token verification ────────────────────────────────────────────────────────

def verify_challenge_token(token: str, secret_key: str) -> tuple[bool, str]:
    """
    Verify signature, TTL, and one-time use.
    Returns (True, nonce) on success; (False, reason) on any failure.
    Consuming the nonce is atomic under _lock.
    """
    try:
        payload_b64, sig = token.rsplit(".", 1)
    except ValueError:
        return False, "Malformed token"

    expected = hmac.new(secret_key.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False, "Invalid signature"

    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode())
        nonce: str = payload["nonce"]
        ts: int    = payload["ts"]
    except Exception:
        return False, "Malformed payload"

    if time.time() - ts > TOKEN_TTL_SECONDS:
        return False, "Token expired"

    with _lock:
        _purge_expired()
        if nonce in _used_nonces:
            return False, "Token already used"
        # Consume: store until natural expiry so replays are rejected
        _used_nonces[nonce] = ts + TOKEN_TTL_SECONDS

    return True, nonce


# ── Frame motion verification ─────────────────────────────────────────────────

def verify_frame_motion(frame_bytes_list: list[bytes]) -> tuple[bool, str]:
    """
    Confirm that the 3 submitted webcam frames are genuinely distinct.

    Why 3 frames:
      frame_0  →  neutral position (face centred)
      frame_1  →  captured after the left-turn motion phase
      frame_2  →  captured after the right-turn motion phase

    An attacker submitting the same static image three times will produce
    near-zero pixel diffs and be rejected. A real head turn shifts the face
    enough to produce a mean absolute diff well above MOTION_THRESHOLD.

    Resizes to 120×120 for speed; the face occupies most of the webcam view
    so the crop is representative.
    """
    if len(frame_bytes_list) != 3:
        return False, "Expected exactly 3 frames"

    arrays: list[np.ndarray] = []
    for i, fb in enumerate(frame_bytes_list):
        try:
            img = Image.open(BytesIO(fb)).convert("RGB").resize((120, 120))
            arrays.append(np.array(img, dtype=np.float32))
        except Exception:
            return False, f"Could not decode frame_{i}"

    diff_01 = float(np.mean(np.abs(arrays[1] - arrays[0])))
    diff_12 = float(np.mean(np.abs(arrays[2] - arrays[1])))

    # Identical frames: attacker submitted the same image multiple times
    if diff_01 < IDENTICAL_THRESHOLD:
        return False, "Frames 0 and 1 are identical — no motion detected"
    if diff_12 < IDENTICAL_THRESHOLD:
        return False, "Frames 1 and 2 are identical — no motion detected"

    # Below-threshold motion: camera shake or minor drift, not a real head turn
    if diff_01 < MOTION_THRESHOLD:
        return False, f"Insufficient motion 0→1 (diff {diff_01:.1f}, need >{MOTION_THRESHOLD})"
    if diff_12 < MOTION_THRESHOLD:
        return False, f"Insufficient motion 1→2 (diff {diff_12:.1f}, need >{MOTION_THRESHOLD})"

    return True, "OK"
