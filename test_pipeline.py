#!/usr/bin/env python3
"""
Script de test complet pour HygieSync
Vérifie toutes les étapes du pipeline avant l'enregistrement vidéo
"""

import os
import sys
import json
import tempfile
import shutil
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_ok(msg: str):
    print(f"  ✓ {msg}")


def print_fail(msg: str):
    print(f"  ✗ {msg}")


def print_info(msg: str):
    print(f"  ℹ {msg}")


def test_imports():
    """Test 1: Vérifier les imports"""
    print_header("TEST 1: Imports")
    
    errors = []
    
    try:
        import torch
        print_ok(f"PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
    
    try:
        import cv2
        print_ok(f"OpenCV {cv2.__version__}")
    except ImportError as e:
        errors.append(f"OpenCV: {e}")
    
    try:
        import mediapipe as mp
        print_ok(f"MediaPipe {mp.__version__}")
    except ImportError as e:
        errors.append(f"MediaPipe: {e}")
    
    try:
        import librosa
        print_ok(f"Librosa {librosa.__version__}")
    except ImportError as e:
        errors.append(f"Librosa: {e}")
    
    try:
        import numpy as np
        print_ok(f"NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"NumPy: {e}")
    
    try:
        from src.config import IMG_SIZE, SR, DEVICE
        print_ok("Config module")
    except ImportError as e:
        errors.append(f"Config: {e}")
    
    try:
        from src.model import HygieUNetLite
        print_ok("Model module")
    except ImportError as e:
        errors.append(f"Model: {e}")
    
    try:
        from src.dataset import MouthDataset
        print_ok("Dataset module")
    except ImportError as e:
        errors.append(f"Dataset: {e}")
    
    try:
        from src.losses import weighted_l1, temporal_l1
        print_ok("Losses module")
    except ImportError as e:
        errors.append(f"Losses: {e}")
    
    if errors:
        for e in errors:
            print_fail(e)
        return False
    return True


def test_gpu():
    """Test 2: Vérifier GPU"""
    print_header("TEST 2: GPU / CUDA")
    
    import torch
    
    if torch.cuda.is_available():
        print_ok(f"CUDA available")
        print_ok(f"Device: {torch.cuda.get_device_name(0)}")
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_ok(f"VRAM: {total_mem:.1f} GB")
        
        x = torch.randn(1, 3, 128, 128).cuda()
        y = x * 2
        print_ok("CUDA tensor operations work")
        return True
    else:
        print_info("CUDA not available, will use CPU")
        return True


def test_model():
    """Test 3: Vérifier le modèle"""
    print_header("TEST 3: Model Architecture")
    
    from src.model import HygieUNetLite
    from src.config import IMG_SIZE, MEL_WIN, DEVICE
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    
    net = HygieUNetLite().to(device)
    
    params = sum(p.numel() for p in net.parameters())
    print_ok(f"Model parameters: {params:,}")
    
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
    mel = torch.randn(2, 1, 80, MEL_WIN).to(device)
    
    with torch.no_grad():
        y = net(x, mel)
    
    print_ok(f"Input shape: {list(x.shape)}")
    print_ok(f"Audio shape: {list(mel.shape)}")
    print_ok(f"Output shape: {list(y.shape)}")
    
    assert y.shape == x.shape, "Output shape mismatch"
    print_ok("Forward pass successful")
    
    return True


def test_losses():
    """Test 4: Vérifier les losses"""
    print_header("TEST 4: Loss Functions")
    
    from src.losses import weighted_l1, temporal_l1
    
    yhat = torch.randn(4, 3, 128, 128, requires_grad=True)
    y = torch.randn(4, 3, 128, 128)
    mask = torch.rand(4, 1, 128, 128)
    
    loss1 = weighted_l1(yhat, y, mask)
    print_ok(f"Weighted L1 loss: {loss1.item():.4f}")
    
    yhat_prev = torch.randn(4, 3, 128, 128, requires_grad=True)
    y_prev = torch.randn(4, 3, 128, 128)
    
    loss2 = temporal_l1(yhat, yhat_prev, y, y_prev, mask)
    print_ok(f"Temporal L1 loss: {loss2.item():.4f}")
    
    total = loss1 + 0.5 * loss2
    total.backward()
    print_ok("Backward pass successful")
    
    return True


def test_mediapipe():
    """Test 5: Vérifier MediaPipe"""
    print_header("TEST 5: MediaPipe Face Detection")
    
    import mediapipe as mp
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
    
    LIPS_IDX = sorted({i for a, b in FACEMESH_LIPS for i in (a, b)})
    print_ok(f"Lips indices: {len(LIPS_IDX)} points")
    
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )
    
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 200), 80, (200, 180, 160), -1)
    cv2.ellipse(img, (320, 220), (30, 10), 0, 0, 180, (150, 100, 100), -1)
    
    res = mp_face.process(img)
    mp_face.close()
    
    print_ok("MediaPipe initialized successfully")
    print_info("(Synthetic face not detected - this is expected)")
    
    return True


def test_audio_processing():
    """Test 6: Vérifier le traitement audio"""
    print_header("TEST 6: Audio Processing")
    
    import librosa
    from src.config import SR, N_MELS
    
    duration = 2.0
    t = np.linspace(0, duration, int(SR * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=SR, n_fft=1024, 
        hop_length=int(SR/25), win_length=1024, n_mels=N_MELS, center=False
    )
    mel_db = np.log(np.maximum(mel, 1e-5))
    
    print_ok(f"Audio duration: {duration}s")
    print_ok(f"Sample rate: {SR}")
    print_ok(f"Mel shape: {mel_db.shape}")
    
    return True


def test_video_io():
    """Test 7: Vérifier lecture/écriture vidéo"""
    print_header("TEST 7: Video I/O")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_video = os.path.join(tmpdir, "test.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(test_video, fourcc, 25.0, (640, 480))
        
        for i in range(25):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            vw.write(frame)
        vw.release()
        
        print_ok("Video writing works")
        
        cap = cv2.VideoCapture(test_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print_ok(f"Video reading works: {frames} frames @ {fps} FPS")
    
    return True


def test_ffmpeg():
    """Test 8: Vérifier ffmpeg"""
    print_header("TEST 8: FFmpeg")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, text=True, timeout=5
        )
        version = result.stdout.split('\n')[0]
        print_ok(f"FFmpeg: {version[:50]}...")
        return True
    except FileNotFoundError:
        print_fail("FFmpeg not found")
        return False
    except Exception as e:
        print_fail(f"FFmpeg error: {e}")
        return False


def test_seamless_clone():
    """Test 9: Vérifier seamlessClone"""
    print_header("TEST 9: Seamless Clone (Poisson Blending)")
    
    dst = np.zeros((200, 200, 3), dtype=np.uint8)
    dst[:] = (100, 100, 100)
    
    src = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(src, (100, 100), 30, (0, 0, 255), -1)
    
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 30, 255, -1)
    
    center = (100, 100)
    
    try:
        result = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        print_ok("seamlessClone works")
        return True
    except cv2.error as e:
        print_fail(f"seamlessClone error: {e}")
        return False


def test_end_to_end_mini():
    """Test 10: Mini test end-to-end"""
    print_header("TEST 10: Mini End-to-End Pipeline")
    
    from src.model import HygieUNetLite
    from src.losses import weighted_l1
    from src.config import IMG_SIZE, MEL_WIN, DEVICE
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    
    net = HygieUNetLite().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
    mel = torch.randn(4, 1, 80, MEL_WIN).to(device)
    y = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
    m = torch.rand(4, 1, IMG_SIZE, IMG_SIZE).to(device)
    
    net.train()
    for i in range(3):
        yhat = net(x, mel)
        loss = weighted_l1(yhat, y, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    print_ok(f"Training step loss: {loss.item():.4f}")
    
    net.eval()
    with torch.no_grad():
        yhat = net(x, mel)
    
    print_ok("Inference works")
    print_ok(f"Output range: [{yhat.min().item():.3f}, {yhat.max().item():.3f}]")
    
    return True


def check_data_files():
    """Vérifie les fichiers de données nécessaires"""
    print_header("DATA FILES CHECK")
    
    files_needed = [
        ("data/train_video.mp4", "Training video (5+ min face camera)"),
        ("data/template_idle.mp4", "Template video (person breathing/blinking)"),
        ("data/new_audio.wav", "Audio to sync (for inference)")
    ]
    
    all_present = True
    for path, desc in files_needed:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1e6
            print_ok(f"{path} ({size:.1f} MB)")
        else:
            print_info(f"MISSING: {path}")
            print_info(f"         -> {desc}")
            all_present = False
    
    return all_present


def main():
    print("\n" + "=" * 60)
    print("  HYGIE-SYNC PIPELINE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("GPU", test_gpu),
        ("Model", test_model),
        ("Losses", test_losses),
        ("MediaPipe", test_mediapipe),
        ("Audio", test_audio_processing),
        ("Video I/O", test_video_io),
        ("FFmpeg", test_ffmpeg),
        ("SeamlessClone", test_seamless_clone),
        ("End-to-End", test_end_to_end_mini),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print_fail(f"Exception: {e}")
            results.append((name, False))
    
    data_ready = check_data_files()
    
    print_header("SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Tests: {passed}/{total} passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        if data_ready:
            print("  📹 Ready to record your video!")
            print("\n  NEXT STEPS:")
            print("  1. python -m src.prepare_dataset data/train_video.mp4 data/ds")
            print("  2. python -m src.train data/ds runs/hygie")
            print("  3. python -m src.infer runs/hygie/ckpt_best.pt data/template_idle.mp4 data/new_audio.wav out_sync.mp4")
        else:
            print("\n  ⚠️  Add the required data files, then run the pipeline.")
    else:
        print("\n  ⚠️  Some tests failed. Fix issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
