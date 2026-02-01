"""
Inférence HygieSync - Génère une vidéo lip-sync
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


def infer(
    ckpt: str = "runs/hygie/ckpt_best.pt",
    template_video: str = "data/template_idle.mp4",
    audio_wav: str = "data/new_audio.wav",
    out_video: str = "out_sync.mp4"
):
    """
    Génère une vidéo lip-sync
    
    Args:
        ckpt: Chemin vers le checkpoint du modèle
        template_video: Vidéo template (personne qui respire/cligne)
        audio_wav: Audio à synchroniser
        out_video: Vidéo de sortie
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

    ok, frame0 = cap.read()
    if not ok:
        raise ValueError("Cannot read template video")
    H, W = frame0.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video, fourcc, float(fps), (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_box = None
    t = 0

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    print(f"Processing {n_frames} frames...")

    # --- PRÉPARATION AUDIO (AVANT LA BOUCLE) ---
    # 1. Ajuster la longueur : on veut être sûr d'avoir assez d'audio pour n_frames + MEL_WIN
    target_len = n_frames + MEL_WIN * 2
    if mel.shape[1] < target_len:
        pad_len = target_len - mel.shape[1]
        pad = np.repeat(mel[:, -1:], pad_len, axis=1)
        mel = np.concatenate([mel, pad], axis=1)

    # 2. Padding pour centrer la fenêtre
    pad_w = MEL_WIN // 2
    mel_padded = np.pad(mel, ((0, 0), (pad_w, pad_w)), mode='edge')
    # -------------------------------------------

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        
        if not res.multi_face_landmarks:
            vw.write(frame)
            t += 1
            continue

        lm = res.multi_face_landmarks[0].landmark
        x0, y0, x1, y1, pts = mouth_bbox_from_landmarks(lm, W, H, MOUTH_PAD)

        smoothed = smooth_bbox(prev_box, (x0, y0, x1, y1), EMA_ALPHA)
        prev_box = smoothed
        x0i, y0i, x1i, y1i = clamp_box(*smoothed, W, H)

        if x1i <= x0i or y1i <= y0i:
            vw.write(frame)
            t += 1
            continue

        crop = frame[y0i:y1i, x0i:x1i].copy()
        crop_rs = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        mask, _ = create_mouth_mask(pts, IMG_SIZE, x0i, y0i, x1i, y1i)

        x_masked = crop_rs.copy()
        x_masked[mask > 0] = 0

        # Audio conditioning avec alignement corrigé
        mel_chunk = mel_padded[:, t : t + MEL_WIN]
        
        # Sécurité ultime
        if mel_chunk.shape[1] != MEL_WIN:
             mel_chunk = np.pad(mel_chunk, ((0,0), (0, MEL_WIN - mel_chunk.shape[1])), mode='edge')
        
        mel_chunk = mel_chunk[None, None, ...]

        # IMPORTANT: Convertir BGR -> RGB pour le modèle
        x_rgb = cv2.cvtColor(x_masked, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x_t = torch.from_numpy(x_rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
        a_t = torch.from_numpy(mel_chunk).to(dev)

        with torch.no_grad():
            yhat = net(x_t, a_t)[0].permute(1, 2, 0).cpu().numpy()
        
        # IMPORTANT: Convertir RGB -> BGR pour OpenCV
        yhat_bgr = cv2.cvtColor((yhat * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        center = ((x0i + x1i) // 2, (y0i + y1i) // 2)

        mask_big = cv2.resize(mask, (x1i - x0i, y1i - y0i), interpolation=cv2.INTER_NEAREST)
        src_big = cv2.resize(yhat_bgr, (x1i - x0i, y1i - y0i), interpolation=cv2.INTER_CUBIC)

        src_frame = np.zeros_like(frame)
        src_frame[y0i:y1i, x0i:x1i] = src_big

        mask_frame = np.zeros((H, W), dtype=np.uint8)
        mask_frame[y0i:y1i, x0i:x1i] = mask_big

        try:
            blended = cv2.seamlessClone(src_frame, frame, mask_frame, center, cv2.NORMAL_CLONE)
        except cv2.error:
            blended = frame.copy()
            blended[y0i:y1i, x0i:x1i] = src_big

        vw.write(blended)
        t += 1

        if t % 100 == 0:
            print(f"  Processed {t}/{n_frames} frames")

    cap.release()
    vw.release()
    mp_face.close()
    
    # AJOUTER L'AUDIO avec ffmpeg (optionnel si le user veut le son direct, mais infer.py ne le faisait pas avant. Ajoutons le pour être complet)
    # Pour l'instant infer.py sort juste la video muette si je suis le code original, mais infer_fast ajoutait l'audio.
    # Le user a demandé "avoir la voix". Ajoutons-le.
    
    # Mais attention, infer.py n'utilisait pas de fichier temporaire avant.
    # Modifions pour utiliser un temp et merger.
    
    temp_video = out_video + ".temp.mp4"
    os.rename(out_video, temp_video)
    
    print("Adding audio...")
    cmd = [
        "ffmpeg", "-y", "-i", temp_video, "-i", audio_wav,
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        out_video
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(temp_video)
        print(f"Saved: {out_video}")
    except subprocess.CalledProcessError:
        print("Warning: ffmpeg audio merge failed. Outputting video without audio.")
        os.rename(temp_video, out_video) # Restore original

    return out_video


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    ckpt = args[0] if len(args) > 0 else "runs/hygie/ckpt_best.pt"
    template = args[1] if len(args) > 1 else "data/template_idle.mp4"
    audio = args[2] if len(args) > 2 else "data/new_audio.wav"
    out = args[3] if len(args) > 3 else "out_sync.mp4"
    infer(ckpt, template, audio, out)
