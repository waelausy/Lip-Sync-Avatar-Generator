"""
Inférence HygieSync - Version rapide pour test
Utilise seamlessClone pour un blending naturel
"""

import os
import cv2
import numpy as np
import librosa
import mediapipe as mp
import torch
import subprocess

from .config import IMG_SIZE, SR, N_MELS, MEL_WIN, FPS_FALLBACK, MOUTH_PAD, EMA_ALPHA, DEVICE
from .model import HygieUNetLite
from .utils import mouth_bbox_from_landmarks, clamp_box, smooth_bbox, create_mouth_mask


def infer_fast(
    ckpt: str = "runs/hygie/ckpt_best.pt",
    template_video: str = "data/train_video.mp4",
    audio_wav: str = "data/new_audio.wav",
    out_video: str = "out_sync_fast.mp4",
    max_frames: int = 500  # Limite pour test rapide
):
    """
    Génère une vidéo lip-sync (version rapide pour test)
    """
    dev = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
    print(f"Inference on: {dev}")
    
    net = HygieUNetLite().to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    net.eval()

    y, _ = librosa.load(audio_wav, sr=SR, mono=True)

    cap = cv2.VideoCapture(template_video)
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

    # Récupérer la rotation
    cmd = ["ffmpeg", "-i", template_video]
    rotation = 0
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stderr.split('\n'):
            if "rotate" in line:
                rotation = int(line.split(":")[1].strip())
    except:
        pass
    
    print(f"Detected rotation: {rotation}°")

    ok, frame0 = cap.read()
    if not ok:
        raise ValueError("Cannot read template video")
    
    # Appliquer la rotation sur la première frame pour avoir les dimensions correctes
    if rotation == 90:
        frame0 = cv2.rotate(frame0, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame0 = cv2.rotate(frame0, cv2.ROTATE_180)
    elif rotation == 270:
        frame0 = cv2.rotate(frame0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    H, W = frame0.shape[:2]
    print(f"Output dimensions: {W}x{H}")
    
    # Vidéo temporaire sans audio
    temp_video = "temp_no_audio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(temp_video, fourcc, float(fps), (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_box = None
    t = 0

    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), max_frames)
    audio_frames = mel.shape[1]

    print(f"Processing {n_frames} frames (max {max_frames})...")

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        # Rotation
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        if not res.multi_face_landmarks:
            continue

        lm = res.multi_face_landmarks[0].landmark
        x0, y0, x1, y1, mouth_pts = mouth_bbox_from_landmarks(lm, w, h, MOUTH_PAD)

        smoothed = smooth_bbox(prev_box, (x0, y0, x1, y1), EMA_ALPHA)
        prev_box = smoothed
        x0i, y0i, x1i, y1i = clamp_box(*smoothed, w, h)

        if x1i <= x0i or y1i <= y0i:
            continue

        crop = frame[y0i:y1i, x0i:x1i].copy()
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        mask, _ = create_mouth_mask(mouth_pts, IMG_SIZE, x0i, y0i, x1i, y1i)
        x_masked = crop.copy()
        x_masked[mask > 0] = 0

        # Audio conditioning
        t = min(i, audio_frames - MEL_WIN)
        mel_chunk = mel[:, t:t+MEL_WIN]

        # Forward pass
        x_tensor = torch.from_numpy(x_masked.transpose(2, 0, 1)[np.newaxis, ...] / 255.0).float().to(dev)
        mel_tensor = torch.from_numpy(mel_chunk[np.newaxis, np.newaxis, ...]).float().to(dev)

        with torch.no_grad():
            yhat = net(x_tensor, mel_tensor).cpu().numpy()[0]

        yhat = np.clip(yhat.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        yhat = cv2.resize(yhat, (x1i-x0i, y1i-y0i), interpolation=cv2.INTER_AREA)

        # Seamless clone
        center = (x0i + (x1i-x0i)//2, y0i + (y1i-y0i)//2)
        try:
            blended = cv2.seamlessClone(yhat, frame, mask, center, cv2.MIXED_CLONE)
        except:
            blended = frame.copy()
            blended[y0i:y1i, x0i:x1i] = yhat

        vw.write(blended)

        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{n_frames} frames")

    cap.release()
    mp_face.close()
    vw.release()

    # AJOUTER L'AUDIO avec ffmpeg
    print("Adding audio...")
    cmd = [
        "ffmpeg", "-y", "-i", temp_video, "-i", audio_wav,
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        out_video
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Nettoyer
    os.remove(temp_video)

    print(f"✅ Done! Output: {out_video}")
    return out_video


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    ckpt = args[0] if len(args) > 0 else "runs/hygie/ckpt_best.pt"
    template = args[1] if len(args) > 1 else "data/train_video.mp4"
    audio = args[2] if len(args) > 2 else "data/new_audio.wav"
    out = args[3] if len(args) > 3 else "out_sync_fast.mp4"
    infer_fast(ckpt, template, audio, out)
