import cv2
import mediapipe as mp
import numpy as np
import sys

def check_orientation(video_path):
    print(f"Checking orientation for: {video_path}")
    
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )
    
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    
    if not ok:
        print("❌ Cannot read video")
        return None

    # Indices: Oeil gauche (33), Oeil droit (263), Lèvre haut (13), Lèvre bas (14)
    # On veut (Eye_Y + Eye_Y)/2 < (Lip_Y + Lip_Y)/2  (Les yeux plus haut que la bouche)
    
    rotations = [
        (0, None, "Original"),
        (90, cv2.ROTATE_90_CLOCKWISE, "90° Clockwise"),
        (180, cv2.ROTATE_180, "180°"),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE, "90° Counter-Clockwise (270°)")
    ]
    
    best_rotation = 0
    found = False
    
    for angle, code, name in rotations:
        if code is not None:
            img = cv2.rotate(frame, code)
        else:
            img = frame
            
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        
        status = "No face"
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            
            # Coordonnées Y normalisées (0 en haut, 1 en bas)
            left_eye_y = lm[33].y
            right_eye_y = lm[263].y
            mouth_y = lm[13].y
            
            avg_eyes = (left_eye_y + right_eye_y) / 2
            
            if avg_eyes < mouth_y:
                status = "✅ UPRIGHT (Correct)"
                print(f"Testing {name}: {status}")
                best_rotation = angle
                found = True
                break
            else:
                status = "❌ Upside Down / Sideways"
        
        print(f"Testing {name}: {status}")
        
    mp_face.close()
    
    if found:
        print(f"\n🎉 FOUND CORRECT ROTATION: {best_rotation}°")
        return best_rotation
    else:
        print("\n⚠️ Could not determine orientation (no face found in any rotation?)")
        return 0

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/train_video.mp4"
    check_orientation(video)
