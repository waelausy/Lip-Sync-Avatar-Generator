"""
Fonctions de loss pour HygieSync
- Weighted L1: Focus sur la zone bouche
- Temporal L1: Anti-flicker entre frames
"""

import torch


def weighted_l1(yhat, y, mask, w_in: float = 8.0, w_out: float = 1.0):
    """
    L1 loss pondérée: plus de poids sur la zone bouche
    
    Args:
        yhat: Prédiction [B,3,H,W]
        y: Ground truth [B,3,H,W]
        mask: Masque bouche [B,1,H,W] in [0..1]
        w_in: Poids intérieur bouche
        w_out: Poids extérieur bouche
    """
    w = w_out + (w_in - w_out) * mask
    return (w * (yhat - y).abs()).mean()


def temporal_l1(yhat_t, yhat_prev, y_t, y_prev, mask, alpha: float = 1.0):
    """
    Loss temporelle: pénalise les différences frame-to-frame
    Réduit le flickering dans la zone bouche
    
    Args:
        yhat_t: Prédiction frame t
        yhat_prev: Prédiction frame t-1
        y_t: Ground truth frame t
        y_prev: Ground truth frame t-1
        mask: Masque bouche
        alpha: Coefficient de pondération
    """
    dy_hat = yhat_t - yhat_prev
    dy_gt = y_t - y_prev
    return alpha * ((dy_hat - dy_gt).abs() * mask).mean()
