"""
Script de test pour vérifier la détection des landmarks lèvres
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

LIPS_IDX = sorted({i for a, b in FACEMESH_LIPS for i in (a, b)})


def probe_landmarks(video_path: str, show: bool = True):
    """
    Vérifie que MediaPipe détecte correctement les lèvres
    
    Args:
        video_path: Chemin vers la vidéo
        show: Afficher la visualisation
    
    Returns:
        True si détection OK
    """
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    
    if not ok:
        print(f"ERROR: Cannot read video {video_path}")
        return False

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)
    mp_face.close()
    
    if not res.multi_face_landmarks:
        print("ERROR: No face detected in video")
        return False

    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in LIPS_IDX], dtype=np.int32)

    dbg = frame.copy()
    for (x, y) in pts:
        cv2.circle(dbg, (x, y), 2, (0, 255, 0), -1)

    hull = cv2.convexHull(pts)
    cv2.polylines(dbg, [hull], True, (255, 0, 0), 2)

    print(f"OK: Lips detected with {len(pts)} points")
    print(f"   Frame size: {w}x{h}")
    print(f"   Lips bbox: x=[{pts[:,0].min()}-{pts[:,0].max()}], y=[{pts[:,1].min()}-{pts[:,1].max()}]")

    if show:
        cv2.imwrite("lips_probe_result.png", dbg)
        print("   Saved: lips_probe_result.png")

    return True


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/train_video.mp4"
    probe_landmarks(video)
