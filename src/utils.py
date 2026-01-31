"""
Utilitaires communs pour HygieSync
"""

import subprocess
import numpy as np
import cv2
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

LIPS_IDX = sorted({i for a, b in FACEMESH_LIPS for i in (a, b)})


def extract_audio_ffmpeg(video_path: str, wav_out: str, sr: int = 16000):
    """Extrait l'audio d'une vidéo en WAV mono"""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr), wav_out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mouth_bbox_from_landmarks(lm, w: int, h: int, mouth_pad: float = 0.35):
    """Calcule la bounding box de la bouche à partir des landmarks"""
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in LIPS_IDX], dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0), (y1 - y0)
    s = max(bw, bh) * (1.0 + mouth_pad)

    x0 = cx - s / 2
    y0 = cy - s / 2
    x1 = cx + s / 2
    y1 = cy + s / 2
    return x0, y0, x1, y1, pts


def clamp_box(x0: float, y0: float, x1: float, y1: float, w: int, h: int):
    """Clamp la bounding box aux dimensions de l'image"""
    x0 = int(max(0, min(w - 1, x0)))
    y0 = int(max(0, min(h - 1, y0)))
    x1 = int(max(1, min(w, x1)))
    y1 = int(max(1, min(h, y1)))
    return x0, y0, x1, y1


def smooth_bbox(prev_box, new_box, alpha: float = 0.7):
    """Lissage EMA de la bounding box"""
    if prev_box is None:
        return new_box
    px0, py0, px1, py1 = prev_box
    x0, y0, x1, y1 = new_box
    sx0 = alpha * px0 + (1 - alpha) * x0
    sy0 = alpha * py0 + (1 - alpha) * y0
    sx1 = alpha * px1 + (1 - alpha) * x1
    sy1 = alpha * py1 + (1 - alpha) * y1
    return sx0, sy0, sx1, sy1


def create_mouth_mask(mouth_pts: np.ndarray, img_size: int, x0i: int, y0i: int, x1i: int, y1i: int):
    """Crée un masque de la bouche dans l'espace crop"""
    pts_px = mouth_pts.copy()
    pts_px[:, 0] = (pts_px[:, 0] - x0i) / max(1, (x1i - x0i)) * img_size
    pts_px[:, 1] = (pts_px[:, 1] - y0i) / max(1, (y1i - y0i)) * img_size
    pts_px = pts_px.astype(np.int32)
    
    hull = cv2.convexHull(pts_px)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask, hull
