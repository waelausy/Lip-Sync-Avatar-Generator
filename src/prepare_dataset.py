"""
Préparation du dataset pour HygieSync
Extrait frames, audio aligné et masques bouche
"""

import os
import json
import cv2
import numpy as np
import librosa
import mediapipe as mp
from tqdm import tqdm

from .config import IMG_SIZE, SR, N_MELS, MOUTH_PAD, EMA_ALPHA, FPS_FALLBACK
from .utils import (
    extract_audio_ffmpeg, 
    mouth_bbox_from_landmarks, 
    clamp_box, 
    smooth_bbox,
    create_mouth_mask
)


import subprocess

def get_video_rotation(video_path: str):
    """Récupère la rotation de la vidéo via ffmpeg"""
    cmd = ["ffmpeg", "-i", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stderr.split('\n'):
        if "rotate" in line:
            try:
                # Format: "rotate          : 90"
                return int(line.split(":")[1].strip())
            except:
                pass
    return 0

def rotate_frame_if_needed(frame: np.ndarray, rotation: int):
    """Corrige l'orientation de la frame selon les métadonnées"""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def prepare_dataset(video_path: str, out_dir: str):
    """
    Prépare le dataset à partir d'une vidéo d'entraînement
    
    Args:
        video_path: Chemin vers la vidéo d'entraînement
        out_dir: Répertoire de sortie pour le dataset
    """
    rotation = get_video_rotation(video_path)
    print(f"Detected video rotation: {rotation}°")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "X_masked"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "Y"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "M"), exist_ok=True)

    wav_path = os.path.join(out_dir, "audio.wav")
    extract_audio_ffmpeg(video_path, wav_path, SR)
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = FPS_FALLBACK

    hop = int(SR / fps)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024, hop_length=hop, win_length=1024,
        n_mels=N_MELS, center=False
    )
    mel = np.log(np.maximum(mel, 1e-5)).astype(np.float32)

    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )

    prev_box = None
    meta = {"fps": float(fps), "sr": SR, "hop": int(hop), "img_size": IMG_SIZE}

    idx = 0
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    skipped = 0

    for _ in tqdm(range(frames_count if frames_count > 0 else 10**9), desc="Extracting"):
        ok, frame = cap.read()
        if not ok:
            break

        # CORRECTION ORIENTATION
        frame = rotate_frame_if_needed(frame, rotation)
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        
        if not res.multi_face_landmarks:
            skipped += 1
            continue

        lm = res.multi_face_landmarks[0].landmark
        x0, y0, x1, y1, mouth_pts = mouth_bbox_from_landmarks(lm, w, h, MOUTH_PAD)

        smoothed = smooth_bbox(prev_box, (x0, y0, x1, y1), EMA_ALPHA)
        prev_box = smoothed
        x0i, y0i, x1i, y1i = clamp_box(*smoothed, w, h)

        if x1i <= x0i or y1i <= y0i:
            skipped += 1
            continue

        crop = frame[y0i:y1i, x0i:x1i].copy()
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        mask, _ = create_mouth_mask(mouth_pts, IMG_SIZE, x0i, y0i, x1i, y1i)

        x_masked = crop.copy()
        x_masked[mask > 0] = 0

        cv2.imwrite(os.path.join(out_dir, "X_masked", f"{idx:06d}.png"), x_masked)
        cv2.imwrite(os.path.join(out_dir, "Y", f"{idx:06d}.png"), crop)
        cv2.imwrite(os.path.join(out_dir, "M", f"{idx:06d}.png"), mask)
        idx += 1

    cap.release()
    mp_face.close()

    Tm = mel.shape[1]
    if Tm < idx:
        pad = np.repeat(mel[:, -1:], idx - Tm, axis=1)
        mel = np.concatenate([mel, pad], axis=1)
    else:
        mel = mel[:, :idx]

    np.save(os.path.join(out_dir, "mel.npy"), mel)
    meta["n_frames"] = int(idx)
    meta["skipped_frames"] = int(skipped)
    
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset ready: frames={idx}, mel={mel.shape}, skipped={skipped}")
    return idx


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/train_video.mp4"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/ds"
    prepare_dataset(video, out)
