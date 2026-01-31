"""
Script d'entraînement pour HygieSync
Split 80/20, checkpoints, preview
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils

from .config import DEVICE, BATCH_SIZE, NUM_WORKERS, EPOCHS, LR, CHECKPOINT_INTERVAL, IMG_SIZE
from .dataset import MouthDataset
from .model import HygieUNetLite
from .losses import weighted_l1, temporal_l1

# Automatic Mixed Precision pour accélération
from torch.cuda.amp import autocast, GradScaler


def train(ds_dir: str = "data/ds", out_dir: str = "runs/hygie"):
    """
    Entraîne le modèle HygieSync
    
    Args:
        ds_dir: Répertoire du dataset
        out_dir: Répertoire de sortie pour checkpoints
    """
    os.makedirs(out_dir, exist_ok=True)

    train_ds = MouthDataset(ds_dir, "train")
    val_ds = MouthDataset(ds_dir, "val")

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        raise ValueError("Dataset vide! Vérifiez la préparation des données.")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )

    dev = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
    print(f"Training on: {dev}")

    net = HygieUNetLite().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    
    # AMP pour entraînement plus rapide
    scaler = GradScaler() if dev.type == 'cuda' else None
    print(f"AMP enabled: {scaler is not None}")

    step = 0
    prev_batch = None
    best_vloss = float('inf')

    for epoch in range(EPOCHS):
        net.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        
        for x, mel, y, m in pbar:
            x, mel, y, m = x.to(dev), mel.to(dev), y.to(dev), m.to(dev)

            # AMP forward pass
            if scaler is not None:
                with autocast():
                    yhat = net(x, mel)
                    loss = weighted_l1(yhat, y, m)

                    if prev_batch is not None:
                        x0, mel0, y0, m0, yhat0 = prev_batch
                        if yhat0.shape == yhat.shape:
                            loss = loss + temporal_l1(yhat, yhat0, y, y0, m, alpha=0.5)
                
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                yhat = net(x, mel)
                loss = weighted_l1(yhat, y, m)

                if prev_batch is not None:
                    x0, mel0, y0, m0, yhat0 = prev_batch
                    if yhat0.shape == yhat.shape:
                        loss = loss + temporal_l1(yhat, yhat0, y, y0, m, alpha=0.5)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

            prev_batch = (x.detach(), mel.detach(), y.detach(), m.detach(), yhat.detach())

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1

            if step % CHECKPOINT_INTERVAL == 0:
                grid = torch.cat([x[:8], yhat[:8], y[:8]], dim=0)
                vutils.save_image(grid, os.path.join(out_dir, f"preview_{step:06d}.png"), nrow=8)
                torch.save(net.state_dict(), os.path.join(out_dir, "ckpt_last.pt"))

        net.eval()
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for x, mel, y, m in val_loader:
                x, mel, y, m = x.to(dev), mel.to(dev), y.to(dev), m.to(dev)
                if scaler is not None:
                    with autocast():
                        yhat = net(x, mel)
                        vloss += weighted_l1(yhat, y, m).item()
                else:
                    yhat = net(x, mel)
                    vloss += weighted_l1(yhat, y, m).item()
                n += 1
        vloss = vloss / max(1, n)
        
        scheduler.step(vloss)
        
        print(f"Epoch {epoch} | Train loss: {epoch_loss/len(train_loader):.4f} | Val loss: {vloss:.4f}")
        
        torch.save(net.state_dict(), os.path.join(out_dir, f"ckpt_epoch_{epoch:03d}.pt"))
        
        if vloss < best_vloss:
            best_vloss = vloss
            torch.save(net.state_dict(), os.path.join(out_dir, "ckpt_best.pt"))
            print(f"  -> New best model saved!")

    print(f"Training complete. Best val loss: {best_vloss:.4f}")
    return best_vloss


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "data/ds"
    out = sys.argv[2] if len(sys.argv) > 2 else "runs/hygie"
    train(ds, out)
