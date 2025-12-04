# enroll_simple.py
# Simple enroll: press 'c' to capture image, press 'q' when done.
import cv2, sqlite3, pickle, numpy as np, time
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

DB = "attendance.db"
DEVICE = 'cpu'  # keep simple

# face detector and embedder
mtcnn = MTCNN(keep_all=False, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

def connect():
    return sqlite3.connect(DB)

def enroll(name, num_samples=6):
    cap = cv2.VideoCapture(0)  # 0 is the default camera. Change to RTSP url if needed.
    print("Camera open. Position the person's face. Press 'c' to capture a photo. Need", num_samples, "photos.")
    samples = []
    while len(samples) < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Camera not giving frames. Check camera.")
            break
        display = frame.copy()
        cv2.putText(display, f"Captured {len(samples)}/{num_samples} - press c", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Enroll", display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            samples.append(frame.copy())
            print("Captured sample", len(samples))
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    tensors = []
    for img in samples:
        # MTCNN returns aligned face tensor
        face = mtcnn(img)  # returns a tensor or None
        if face is not None:
            tensors.append(face)
    if not tensors:
        print("No faces detected in the captured photos.")
        return

    # stack and get embeddings
    stack = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        embeddings = resnet(stack).cpu().numpy()

    mean_emb = embeddings.mean(axis=0)
    # store in DB
    conn = connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO students (name, embedding) VALUES (?, ?)", (name, pickle.dumps(mean_emb)))
    conn.commit()
    conn.close()
    print("Enrolled:", name)

if __name__ == "__main__":
    create_name = input("Enter student name (type and press Enter): ").strip()
    if create_name:
        enroll(create_name, num_samples=6)
    else:
        print("No name given. Exiting.")
