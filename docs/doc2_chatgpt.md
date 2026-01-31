Tu as raison : une seule image de référence + “warp” peut suffire pour valider la chaîne, mais ça n’explique pas un rendu “vivant” (micro-mouvements, respiration, petits changements d’éclairage, clignements).
La clé pour obtenir un rendu naturel sans devoir “apprendre tout le visage” est généralement :

Tu gardes le visage “vivant” depuis une vidéo template (idle / écoute / respiration / clignements).

Tu ne synthétises que la zone bouche (et un peu de peau autour) à partir de l’audio.

Tu recolles proprement (blend) la bouche sur la frame template.

C’est exactement l’esprit des approches type Wav2Lip (papier + repo), qui synchronisent la bouche à un audio en conservant l’identité et les mouvements globaux de la vidéo d’entrée.

Ci-dessous, je te donne un plan de route complet, exécutable (projet “from scratch”), avec :

data prep robuste (MediaPipe → face tracking + masque bouche),

dataset PyTorch propre (split 80/20 sans fuite),

modèle “Audio-conditioned U-Net Lite” (conçu pour inférence CPU),

losses (bouche pondérée + temporelle),

inférence + blending (rapide + qualité via seamlessClone OpenCV),

une option “turbo” : partir de Wav2Lip pré-entraîné au lieu d’entraîner from scratch.

0) Ressources internet utiles (déjà “validées”)

Tu pourras les garder en onglets pendant que tu codes :

Wav2Lip (repo + paper) : pipeline lip-sync de référence, et surtout des idées de “chunk audio ↔ frame vidéo”.

MediaPipe FaceMesh / Face Landmarker (468 landmarks temps réel, mobile-friendly).

Comment récupérer proprement les indices lèvres via FACEMESH_LIPS (pas de liste “au pif”).

OpenCV seamlessClone / Poisson blending (collage bouche sans couture).

Delaunay triangulation (si tu veux ensuite faire du warping triangle par triangle).

FAISS (si tu ajoutes plus tard un mode “retrieval HD” pour dents/langue zéro flou).

1) Architecture cible (pro, simple, et “vivante”)
Ce que tu entraînes

Un modèle qui fait :
(frame template avec bouche masquée) + (audio mel chunk) → patch bouche réaliste.

Ce que tu ne réinventes pas

Les clignements / respiration / micro-mouvements : viennent du template vidéo.

La géométrie face globale : vient du template vidéo.

Le modèle n’a pas besoin de “réapprendre tout le visage”, seulement la zone qui parle.

C’est le meilleur compromis “qualité perçue” / “complexité” / “inférence CPU”.

2) Arborescence projet (copie-colle)

Crée un dossier :

hygie_sync/
  requirements.txt
  config.py
  00_probe_landmarks.py
  01_prepare_dataset.py
  dataset.py
  model.py
  losses.py
  train.py
  infer.py
  export_onnx.py   (optionnel)
  data/
    train_video.mp4
    template_idle.mp4

3) Installation (propre)
requirements.txt
numpy
opencv-python
mediapipe
librosa
soundfile
torch
torchvision
tqdm

Install
pip install -r requirements.txt


⚠️ Installe ffmpeg (indispensable pour extraire l’audio correctement ; Wav2Lip l’utilise aussi).

4) Config centrale
config.py
IMG_SIZE = 128          # patch bouche (monte à 192 si tu as un GPU)
SR = 16000
N_MELS = 80
MEL_WIN = 16            # contexte temporel (coarticulation)
TRAIN_SPLIT = 0.8       # 80/20
MOUTH_PAD = 0.35        # marge autour de la bouche
EMA_ALPHA = 0.7         # lissage bbox (réduit jitter)
DEVICE = "cuda"         # "cpu" si pas de GPU
FPS_FALLBACK = 25

5) Étape A — Vérifier tes landmarks + indices lèvres (zéro ambiguïté)
Pourquoi ?

Tu veux être sûr à 100% que :

MediaPipe détecte bien,

ton ROI bouche est stable,

ton masque bouche est bien placé.

00_probe_landmarks.py
import cv2
import mediapipe as mp
import numpy as np

# Astuce: FACEMESH_LIPS donne des connexions (edges). On en tire les indices.
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

VIDEO = "data/train_video.mp4"

LIPS_IDX = sorted({i for a, b in FACEMESH_LIPS for i in (a, b)})

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()
assert ok, "Impossible de lire la vidéo"

h, w = frame.shape[:2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
res = mp_face.process(rgb)
assert res.multi_face_landmarks, "Aucun visage détecté"

lm = res.multi_face_landmarks[0].landmark
pts = np.array([(lm[i].x * w, lm[i].y * h) for i in LIPS_IDX], dtype=np.int32)

# Visualiser points lèvres
dbg = frame.copy()
for (x, y) in pts:
    cv2.circle(dbg, (x, y), 1, (0, 255, 0), -1)

# Hull (masque bouche robuste)
hull = cv2.convexHull(pts)
cv2.polylines(dbg, [hull], True, (255, 0, 0), 1)

cv2.imshow("lips_probe", dbg)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("OK: lèvres détectées et hull tracé.")


Pourquoi c’est fiable ?
Parce que tu ne “hardcodes” pas une liste fragile : tu pars des connexions officielles lèvres.

6) Étape B — Préparer dataset complet (frames + audio aligné + masques)
Objectif dataset

Pour chaque frame t :

X_img[t] : crop bouche avec bouche masquée

X_mel[t] : chunk mel centré autour de t

Y_img[t] : crop bouche ground truth

mask[t] : masque bouche (poids de loss + blending)

01_prepare_dataset.py
import os, json, subprocess
import cv2
import mediapipe as mp
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from config import IMG_SIZE, SR, N_MELS, MEL_WIN, MOUTH_PAD, EMA_ALPHA, FPS_FALLBACK

from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
LIPS_IDX = sorted({i for a, b in FACEMESH_LIPS for i in (a, b)})

def extract_audio_ffmpeg(video_path: str, wav_out: str):
    # ffmpeg -i in.mp4 -ac 1 -ar 16000 out.wav
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(SR), wav_out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def mouth_bbox_from_landmarks(lm, w, h):
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in LIPS_IDX], dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)

    cx, cy = (x0+x1)/2, (y0+y1)/2
    bw, bh = (x1-x0), (y1-y0)
    s = max(bw, bh) * (1.0 + MOUTH_PAD)

    # carré autour de la bouche
    x0 = cx - s/2
    y0 = cy - s/2
    x1 = cx + s/2
    y1 = cy + s/2
    return x0, y0, x1, y1, pts

def clamp_box(x0, y0, x1, y1, w, h):
    x0 = int(max(0, min(w-1, x0)))
    y0 = int(max(0, min(h-1, y0)))
    x1 = int(max(1, min(w,   x1)))
    y1 = int(max(1, min(h,   y1)))
    return x0, y0, x1, y1

def main(video_path="data/train_video.mp4", out_dir="data/ds"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "X_masked"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "Y"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "M"), exist_ok=True)

    wav_path = os.path.join(out_dir, "audio.wav")
    extract_audio_ffmpeg(video_path, wav_path)
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = FPS_FALLBACK

    # Mel aligné sur fps : hop ≈ SR/fps
    hop = int(SR / fps)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024, hop_length=hop, win_length=1024,
        n_mels=N_MELS, center=False
    )
    mel = np.log(np.maximum(mel, 1e-5)).astype(np.float32)  # [80, T]

    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )

    prev_box = None
    meta = {"fps": float(fps), "sr": SR, "hop": int(hop), "img_size": IMG_SIZE}

    idx = 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    for _ in tqdm(range(frames if frames > 0 else 10**9), desc="Extract"):
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if not res.multi_face_landmarks:
            continue

        lm = res.multi_face_landmarks[0].landmark
        x0, y0, x1, y1, mouth_pts = mouth_bbox_from_landmarks(lm, w, h)

        # EMA smoothing bbox (stabilité)
        if prev_box is None:
            sx0, sy0, sx1, sy1 = x0, y0, x1, y1
        else:
            px0, py0, px1, py1 = prev_box
            sx0 = EMA_ALPHA * px0 + (1-EMA_ALPHA) * x0
            sy0 = EMA_ALPHA * py0 + (1-EMA_ALPHA) * y0
            sx1 = EMA_ALPHA * px1 + (1-EMA_ALPHA) * x1
            sy1 = EMA_ALPHA * py1 + (1-EMA_ALPHA) * y1

        prev_box = (sx0, sy0, sx1, sy1)
        x0i, y0i, x1i, y1i = clamp_box(sx0, sy0, sx1, sy1, w, h)

        crop = frame[y0i:y1i, x0i:x1i].copy()      # BGR
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Construire masque bouche dans l’espace crop
        # 1) passer points en coords crop
        mouth_pts_px = mouth_pts.copy()
        mouth_pts_px[:, 0] = (mouth_pts_px[:, 0] - x0i) / max(1, (x1i-x0i)) * IMG_SIZE
        mouth_pts_px[:, 1] = (mouth_pts_px[:, 1] - y0i) / max(1, (y1i-y0i)) * IMG_SIZE
        mouth_pts_px = mouth_pts_px.astype(np.int32)

        hull = cv2.convexHull(mouth_pts_px)
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # X = crop avec bouche masquée
        x_masked = crop.copy()
        x_masked[mask > 0] = 0

        cv2.imwrite(os.path.join(out_dir, "X_masked", f"{idx:06d}.png"), x_masked)
        cv2.imwrite(os.path.join(out_dir, "Y",        f"{idx:06d}.png"), crop)
        cv2.imwrite(os.path.join(out_dir, "M",        f"{idx:06d}.png"), mask)
        idx += 1

    cap.release()

    # Aligner mel sur nb d’images
    # mel est [80, Tm] — on veut Tm >= idx (sinon pad)
    Tm = mel.shape[1]
    if Tm < idx:
        pad = np.repeat(mel[:, -1:], idx - Tm, axis=1)
        mel = np.concatenate([mel, pad], axis=1)
    else:
        mel = mel[:, :idx]

    np.save(os.path.join(out_dir, "mel.npy"), mel)  # [80, T]
    meta["n_frames"] = int(idx)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"OK dataset: frames={idx}, mel={mel.shape}")

if __name__ == "__main__":
    main()


Notes importantes :

MediaPipe FaceMesh estime 468 landmarks et est pensé temps réel.

seamlessClone (Poisson) est la référence pour recoller sans couture.

7) Dataset PyTorch (fenêtres audio propres + split 80/20 sans fuite)
dataset.py
import os, json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from config import MEL_WIN, TRAIN_SPLIT

class MouthDataset(Dataset):
    def __init__(self, ds_dir: str, mode: str):
        self.ds_dir = ds_dir
        self.mode = mode

        meta = json.load(open(os.path.join(ds_dir, "meta.json"), "r", encoding="utf-8"))
        self.n = meta["n_frames"]
        self.mel = np.load(os.path.join(ds_dir, "mel.npy"))  # [80, T]

        split = int(self.n * TRAIN_SPLIT)
        if mode == "train":
            self.i0, self.i1 = 0, split
        else:
            self.i0, self.i1 = split, self.n

    def __len__(self):
        # on évite les bords pour mel_win
        return max(0, (self.i1 - self.i0) - MEL_WIN)

    def __getitem__(self, k):
        t = self.i0 + k + MEL_WIN//2

        x_path = os.path.join(self.ds_dir, "X_masked", f"{t:06d}.png")
        y_path = os.path.join(self.ds_dir, "Y",        f"{t:06d}.png")
        m_path = os.path.join(self.ds_dir, "M",        f"{t:06d}.png")

        x = cv2.imread(x_path)  # BGR
        y = cv2.imread(y_path)
        m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

        # BGR->RGB, [0..1]
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        m = (m.astype(np.float32) / 255.0)[None, ...]  # [1,H,W]

        # audio chunk centré
        a0 = t - MEL_WIN//2
        a1 = a0 + MEL_WIN
        mel_chunk = self.mel[:, a0:a1]   # [80, MEL_WIN]
        mel_chunk = mel_chunk[None, ...] # [1,80,MEL_WIN]

        # to tensor: [C,H,W]
        x = torch.from_numpy(x).permute(2,0,1)
        y = torch.from_numpy(y).permute(2,0,1)
        m = torch.from_numpy(m)
        mel_chunk = torch.from_numpy(mel_chunk)

        return x, mel_chunk, y, m

8) Modèle “Audio-conditioned U-Net Lite” (conçu pour CPU)

Idée :

U-Net visuel (petit),

encodeur audio,

conditionnement audio via FiLM (scale/shift) dans le décodeur.

model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class AudioEncoder(nn.Module):
    # input: [B,1,80,MEL_WIN]
    def __init__(self, emb=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, emb)

    def forward(self, a):
        x = self.net(a).flatten(1)
        return self.fc(x)

class FiLM(nn.Module):
    def __init__(self, emb, channels):
        super().__init__()
        self.to_gamma = nn.Linear(emb, channels)
        self.to_beta  = nn.Linear(emb, channels)

    def forward(self, feat, aemb):
        # feat: [B,C,H,W]
        g = self.to_gamma(aemb)[:, :, None, None]
        b = self.to_beta(aemb)[:, :, None, None]
        return feat * (1.0 + g) + b

class HygieUNetLite(nn.Module):
    def __init__(self, emb=256):
        super().__init__()
        self.aenc = AudioEncoder(emb=emb)

        # Encoder
        self.e1 = conv(3, 32)          # 128
        self.e2 = conv(32, 64, s=2)    # 64
        self.e3 = conv(64, 128, s=2)   # 32
        self.e4 = conv(128, 256, s=2)  # 16

        # Bottleneck
        self.b  = conv(256, 256)

        # Decoder
        self.f4 = FiLM(emb, 256)
        self.d3 = conv(256+128, 128)
        self.f3 = FiLM(emb, 128)
        self.d2 = conv(128+64, 64)
        self.f2 = FiLM(emb, 64)
        self.d1 = conv(64+32, 32)
        self.f1 = FiLM(emb, 32)

        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x_masked, mel_chunk):
        aemb = self.aenc(mel_chunk)

        e1 = self.e1(x_masked)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        b  = self.b(e4)

        # up 16->32
        u3 = F.interpolate(self.f4(b, aemb), scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.d3(torch.cat([u3, e3], dim=1))

        # up 32->64
        u2 = F.interpolate(self.f3(u3, aemb), scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.d2(torch.cat([u2, e2], dim=1))

        # up 64->128
        u1 = F.interpolate(self.f2(u2, aemb), scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.d1(torch.cat([u1, e1], dim=1))

        u1 = self.f1(u1, aemb)

        # Résiduel: on prédit une correction et on l’ajoute au masked input
        delta = torch.tanh(self.out(u1))
        yhat = torch.clamp(x_masked + delta, 0.0, 1.0)
        return yhat

9) Losses (netteté bouche + anti-flicker)
losses.py
import torch
import torch.nn.functional as F

def weighted_l1(yhat, y, mask, w_in=8.0, w_out=1.0):
    # mask: [B,1,H,W] in [0..1]
    w = w_out + (w_in - w_out) * mask
    return (w * (yhat - y).abs()).mean()

def temporal_l1(yhat_t, yhat_prev, y_t, y_prev, mask, alpha=1.0):
    # compare frame-to-frame differences inside mouth to reduce flicker
    dy_hat = (yhat_t - yhat_prev)
    dy_gt  = (y_t - y_prev)
    return alpha * ((dy_hat - dy_gt).abs() * mask).mean()

10) Training (split 80/20, checkpoints, preview)
train.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils

from config import DEVICE
from dataset import MouthDataset
from model import HygieUNetLite
from losses import weighted_l1, temporal_l1

def main(ds_dir="data/ds", out_dir="runs/hygie"):
    os.makedirs(out_dir, exist_ok=True)

    train_ds = MouthDataset(ds_dir, "train")
    val_ds   = MouthDataset(ds_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    dev = torch.device(DEVICE if torch.cuda.is_available() and DEVICE=="cuda" else "cpu")

    net = HygieUNetLite().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)

    step = 0
    prev_batch = None

    for epoch in range(50):
        net.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for x, mel, y, m in pbar:
            x, mel, y, m = x.to(dev), mel.to(dev), y.to(dev), m.to(dev)

            yhat = net(x, mel)
            loss = weighted_l1(yhat, y, m)

            # temporal: on l’active si on a un batch précédent
            if prev_batch is not None:
                x0, mel0, y0, m0, yhat0 = prev_batch
                # même batch shape -> approx ok; sinon ignore
                if yhat0.shape == yhat.shape:
                    loss = loss + temporal_l1(yhat, yhat0, y, y0, m, alpha=0.5)

            opt.zero_grad()
            loss.backward()
            opt.step()

            prev_batch = (x.detach(), mel.detach(), y.detach(), m.detach(), yhat.detach())

            pbar.set_postfix(loss=float(loss.item()))
            step += 1

            if step % 200 == 0:
                # preview
                grid = torch.cat([x[:8], yhat[:8], y[:8]], dim=0)
                vutils.save_image(grid, os.path.join(out_dir, f"preview_{step:06d}.png"), nrow=8)
                torch.save(net.state_dict(), os.path.join(out_dir, "ckpt_last.pt"))

        # mini val
        net.eval()
        with torch.no_grad():
            vloss = 0.0
            n = 0
            for x, mel, y, m in val_loader:
                x, mel, y, m = x.to(dev), mel.to(dev), y.to(dev), m.to(dev)
                yhat = net(x, mel)
                vloss += float(weighted_l1(yhat, y, m).item())
                n += 1
            vloss /= max(1, n)
        print("VAL loss:", vloss)
        torch.save(net.state_dict(), os.path.join(out_dir, f"ckpt_epoch_{epoch:03d}.pt"))

if __name__ == "__main__":
    main()

11) Inférence “vivante” + blending propre (template vidéo, pas image fixe)

Tu prends template_idle.mp4 (la personne respire/cligne).
Pour chaque frame template :

tu extrais le crop bouche + masque (exactement comme dataset),

tu génères la bouche,

tu “recolles” dans la frame d’origine.

infer.py (version pragmatique)
import os, json, subprocess
import cv2
import numpy as np
import librosa
import mediapipe as mp
import torch

from config import IMG_SIZE, SR, N_MELS, MEL_WIN, FPS_FALLBACK, MOUTH_PAD, EMA_ALPHA, DEVICE
from model import HygieUNetLite
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

LIPS_IDX = sorted({i for a, b in FACEMESH_LIPS for i in (a, b)})

def extract_audio_ffmpeg(video_path: str, wav_out: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(SR), wav_out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def clamp_box(x0, y0, x1, y1, w, h):
    x0 = int(max(0, min(w-1, x0)))
    y0 = int(max(0, min(h-1, y0)))
    x1 = int(max(1, min(w,   x1)))
    y1 = int(max(1, min(h,   y1)))
    return x0, y0, x1, y1

def mouth_bbox(lm, w, h, mouth_pad=0.35):
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in LIPS_IDX], dtype=np.float32)
    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    s = max(x1-x0, y1-y0) * (1.0 + mouth_pad)
    return cx-s/2, cy-s/2, cx+s/2, cy+s/2, pts

def main(
    ckpt="runs/hygie/ckpt_last.pt",
    template_video="data/template_idle.mp4",
    audio_wav="data/new_audio.wav",
    out_video="out_sync.mp4"
):
    dev = torch.device(DEVICE if torch.cuda.is_available() and DEVICE=="cuda" else "cpu")
    net = HygieUNetLite().to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev))
    net.eval()

    # audio -> mel
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
    mel = np.log(np.maximum(mel, 1e-5)).astype(np.float32)  # [80, T]

    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )

    # writer
    ok, frame0 = cap.read()
    assert ok
    H, W = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video, fourcc, float(fps), (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_box = None
    t = 0

    # padding mel to length of video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if mel.shape[1] < n_frames:
        pad = np.repeat(mel[:, -1:], n_frames - mel.shape[1], axis=1)
        mel = np.concatenate([mel, pad], axis=1)
    else:
        mel = mel[:, :n_frames]

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
        x0, y0, x1, y1, pts = mouth_bbox(lm, W, H, mouth_pad=MOUTH_PAD)

        # smooth bbox
        if prev_box is None:
            sx0, sy0, sx1, sy1 = x0, y0, x1, y1
        else:
            px0, py0, px1, py1 = prev_box
            sx0 = EMA_ALPHA*px0 + (1-EMA_ALPHA)*x0
            sy0 = EMA_ALPHA*py0 + (1-EMA_ALPHA)*y0
            sx1 = EMA_ALPHA*px1 + (1-EMA_ALPHA)*x1
            sy1 = EMA_ALPHA*py1 + (1-EMA_ALPHA)*y1
        prev_box = (sx0, sy0, sx1, sy1)
        x0i, y0i, x1i, y1i = clamp_box(sx0, sy0, sx1, sy1, W, H)

        crop = frame[y0i:y1i, x0i:x1i].copy()
        crop_rs = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # mask bouche
        pts_px = pts.copy()
        pts_px[:, 0] = (pts_px[:, 0] - x0i) / max(1, (x1i-x0i)) * IMG_SIZE
        pts_px[:, 1] = (pts_px[:, 1] - y0i) / max(1, (y1i-y0i)) * IMG_SIZE
        pts_px = pts_px.astype(np.int32)
        hull = cv2.convexHull(pts_px)
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        x_masked = crop_rs.copy()
        x_masked[mask > 0] = 0

        # audio chunk
        a0 = max(0, t - MEL_WIN//2)
        a1 = a0 + MEL_WIN
        if a1 > mel.shape[1]:
            a0 = mel.shape[1] - MEL_WIN
            a1 = mel.shape[1]
        mel_chunk = mel[:, a0:a1][None, None, ...]  # [1,1,80,MEL_WIN]

        # tensors
        x_rgb = cv2.cvtColor(x_masked, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x_t = torch.from_numpy(x_rgb).permute(2,0,1).unsqueeze(0).to(dev)
        a_t = torch.from_numpy(mel_chunk).to(dev)

        with torch.no_grad():
            yhat = net(x_t, a_t)[0].permute(1,2,0).cpu().numpy()  # [H,W,3] RGB
        yhat_bgr = cv2.cvtColor((yhat*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Recolle dans crop original via seamlessClone (qualité)
        # OpenCV seamlessClone: Poisson blending :contentReference[oaicite:11]{index=11}
        center = ( (x0i+x1i)//2, (y0i+y1i)//2 )

        # mask en coords frame
        mask_big = cv2.resize(mask, (x1i-x0i, y1i-y0i), interpolation=cv2.INTER_NEAREST)
        src_big  = cv2.resize(yhat_bgr, (x1i-x0i, y1i-y0i), interpolation=cv2.INTER_CUBIC)

        # source image sur fond noir (taille frame)
        src_frame = np.zeros_like(frame)
        src_frame[y0i:y1i, x0i:x1i] = src_big

        mask_frame = np.zeros((H, W), dtype=np.uint8)
        mask_frame[y0i:y1i, x0i:x1i] = mask_big

        blended = cv2.seamlessClone(src_frame, frame, mask_frame, center, cv2.NORMAL_CLONE)
        vw.write(blended)
        t += 1

    cap.release()
    vw.release()
    print("Saved:", out_video)

if __name__ == "__main__":
    main()


Pourquoi seamlessClone est important ?
Parce que c’est du Poisson blending (collage par gradients) : ça évite l’effet “sticker bouche” quand l’éclairage varie.

12) Ce que tu peux améliorer ensuite (sans casser la simplicité)
A) “Qualité Wav2Lip” en 1 étape (recommandé si tu veux aller vite)

Au lieu d’entraîner from scratch, tu peux :

utiliser Wav2Lip pré-entraîné direct,

puis fine-tuner sur ta personne si besoin.

Wav2Lip est spécifiquement conçu pour lip-sync “in the wild” et fournit code + modèles.
Le repo décrit aussi que certains datasets web doivent être sync-corrected avant training (à garder en tête si tu collectes des vidéos au hasard).

B) Ajouter une loss lip-sync “expert” (SyncNet)

Le papier “Out of Time” (Chung & Zisserman) est la base de SyncNet (mesure sync audio/visuel).
Ça sert à punir les frames où la bouche est jolie mais désynchronisée.

C) Ajouter un mode “zéro flou dents/langue” par retrieval

Tu crées une base de patches bouche réels (HD) et tu fais du nearest-neighbor via FAISS.
Ensuite tu ajustes par warping / blending.
C’est très performant en CPU, car FAISS est optimisé pour la similarité.

D) Optimisation CPU

réduire IMG_SIZE (96/128),

export ONNX + quantization (INT8),

éviter seamlessClone si tu vises temps réel strict → alpha blending bord flouté (plus rapide).

13) “Checklist” exécution (ordre exact)

Mets data/train_video.mp4 (5–15 min face caméra, propre, stable).

Lance : python 00_probe_landmarks.py → tu vois les lèvres + hull.

Lance : python 01_prepare_dataset.py → ça crée data/ds/* + mel.npy.

Lance : python train.py → ça écrit runs/hygie/preview_*.png + ckpt_*.pt.

Mets un audio test data/new_audio.wav + une vidéo template data/template_idle.mp4.

Lance : python infer.py → sortie out_sync.mp4.

Point éthique (court mais important)

Tout ce pipeline permet de synthétiser une bouche synchronisée. À utiliser uniquement avec consentement explicite de la personne filmée et pour des cas légitimes (formation, avatar autorisé, etc.).

Contexte : je m’appuie sur ton brief et l’historique que tu as partagé dans ce fichier.