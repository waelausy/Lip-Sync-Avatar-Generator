# HygieSync - Lip Sync Avatar Generator

Système de synchronisation labiale audio-pilotée utilisant un U-Net conditionné par audio.

## 🎯 Objectif

Générer des vidéos où la bouche d'une personne est synchronisée avec un audio donné, en conservant les mouvements naturels (respiration, clignements) de la vidéo template.

## 📋 Prérequis

- **GPU**: 4GB VRAM minimum (testé sur Quadro P620)
- **Python**: 3.8+
- **FFmpeg**: Installé et accessible dans PATH

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📁 Structure des données

```
data/
├── train_video.mp4      # Vidéo d'entraînement (5-15 min, face caméra)
├── template_idle.mp4    # Vidéo template (personne qui respire/cligne)
└── new_audio.wav        # Audio à synchroniser
```

### Conseils pour l'enregistrement

**Pour `train_video.mp4`:**
- Durée: 5-15 minutes
- Face caméra, bon éclairage
- Parler normalement avec des expressions variées
- Éviter les mouvements de tête brusques

**Pour `template_idle.mp4`:**
- Peut être la même vidéo ou une vidéo différente
- Contient les mouvements naturels (respiration, clignements)
- Sera utilisée comme base pour l'inférence

**Pour `new_audio.wav`:**
- Format WAV, mono, 16kHz recommandé
- L'audio que vous voulez synchroniser

## 🧪 Tests

```bash
python run_pipeline.py test
```

Vérifie:
- Imports et dépendances
- GPU/CUDA
- Architecture du modèle
- Fonctions de loss
- MediaPipe
- Traitement audio
- I/O vidéo
- FFmpeg
- SeamlessClone

## 🔧 Utilisation

### Option 1: Pipeline complet

```bash
python run_pipeline.py full
```

### Option 2: Étape par étape

```bash
# 1. Vérifier la détection des landmarks
python run_pipeline.py probe data/train_video.mp4

# 2. Préparer le dataset
python run_pipeline.py prepare data/train_video.mp4 --out data/ds

# 3. Entraîner le modèle
python run_pipeline.py train --ds data/ds --out runs/hygie

# 4. Générer la vidéo
python run_pipeline.py infer \
    --ckpt runs/hygie/ckpt_best.pt \
    --template data/template_idle.mp4 \
    --audio data/new_audio.wav \
    --output out_sync.mp4
```

## 📊 Architecture

```
Audio (Mel Spectrogram) ─┐
                         ├─> HygieUNetLite ─> Patch bouche généré
Image (bouche masquée) ──┘

Le patch est ensuite recollé via seamlessClone (Poisson blending)
```

### Modèle: HygieUNetLite

- **Encodeur visuel**: Conv2D avec downsampling
- **Encodeur audio**: Conv2D + pooling adaptatif
- **Conditionnement**: FiLM (Feature-wise Linear Modulation)
- **Décodeur**: U-Net avec skip connections
- **Sortie**: Résiduel (delta) ajouté à l'entrée masquée

### Losses

- **Weighted L1**: Plus de poids sur la zone bouche (8x)
- **Temporal L1**: Pénalise les différences frame-to-frame (anti-flicker)

## 📈 Paramètres (config.py)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| IMG_SIZE | 128 | Taille du patch bouche |
| BATCH_SIZE | 16 | Optimisé pour 4GB VRAM |
| EPOCHS | 50 | Nombre d'epochs |
| MEL_WIN | 16 | Fenêtre temporelle audio |
| TRAIN_SPLIT | 0.8 | 80% train, 20% val |

## 🎬 Ce que vous devez dire dans votre vidéo d'entraînement

Pour un bon résultat, incluez:

1. **Tous les phonèmes français**: 
   - Voyelles: a, e, i, o, u, ou, on, an, in
   - Consonnes: p, b, t, d, k, g, f, v, s, z, ch, j, m, n, l, r

2. **Phrases variées**:
   - "Bonjour, je m'appelle [votre nom]"
   - "Les chaussettes de l'archiduchesse sont-elles sèches?"
   - "Un chasseur sachant chasser doit savoir chasser sans son chien"
   - Comptez de 1 à 100
   - Lisez un article de journal

3. **Expressions**:
   - Sourire
   - Surprise
   - Concentration
   - Neutre

4. **Durée**: Minimum 5 minutes, idéalement 10-15 minutes

## 🐛 Dépannage

### "No face detected"
- Vérifiez l'éclairage
- Assurez-vous que le visage est bien visible

### "CUDA out of memory"
- Réduisez BATCH_SIZE dans config.py
- Réduisez IMG_SIZE à 96

### Vidéo saccadée
- Augmentez EMA_ALPHA (0.8-0.9)
- Vérifiez que le FPS est cohérent

## 📜 Licence

Usage personnel et éducatif uniquement. Utilisation éthique requise.
