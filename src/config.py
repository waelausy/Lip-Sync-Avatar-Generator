"""
Configuration centrale pour HygieSync - Lip Sync Avatar
Optimisé pour GPU 4GB (Quadro P620)
"""

IMG_SIZE = 128          # Taille du patch bouche (128 optimal pour 4GB GPU)
SR = 16000              # Sample rate audio
N_MELS = 80             # Nombre de bandes Mel
MEL_WIN = 16            # Fenêtre temporelle audio (coarticulation)
TRAIN_SPLIT = 0.8       # 80% train, 20% val
MOUTH_PAD = 0.35        # Marge autour de la bouche
EMA_ALPHA = 0.7         # Lissage bbox (réduit jitter)
DEVICE = "cuda"         # "cuda" ou "cpu"
FPS_FALLBACK = 25       # FPS par défaut si non détecté

BATCH_SIZE = 32         # Augmenté pour utiliser plus de VRAM (4GB disponible)
NUM_WORKERS = 4         # Augmenté pour plus de parallélisme
EPOCHS = 50             # Nombre d'epochs
LR = 2e-4               # Learning rate
EMB_DIM = 256           # Dimension embedding audio

CHECKPOINT_INTERVAL = 200  # Sauvegarder preview tous les N steps
