# recognize_safe.py
import cv2, sqlite3, pickle, numpy as np, time, datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

DB = "attendance.db"
DEVICE = 'cpu'

mtcnn = MTCNN(keep_all=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

def load_known():
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("SELECT id, name, embedding FROM students")
    rows = cur.fetchall(); conn.close()
    known = []
    for sid, name, emb_blob in rows:
        emb = pickle.loads(emb_blob)
        emb = emb / np.linalg.norm(emb)
        known.append((sid, name, emb))
    return known

def has_attendance_today(student_id):
    date_iso = datetime.date.today().isoformat()
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("SELECT id FROM attendance WHERE student_id=? AND date_iso=?", (student_id, date_iso))
    row = cur.fetchone(); conn.close()
    return row is not None

def add_attendance(student_id):
    # only add attendance once per day
    if has_attendance_today(student_id):
        return
    date_iso = datetime.date.today().isoformat()
    now = time.time()
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("INSERT INTO attendance (student_id, date_iso, first_seen_ts, last_seen_ts, count) VALUES (?,?,?,?,?)",
                (student_id, date_iso, now, now, 1))
    conn.commit(); conn.close()
    print("Attendance recorded for id:", student_id)

def cosine_distance(a,b):
    return 1 - np.dot(a,b)  # both normalized

# TUNE THIS:
THRESHOLD = 0.45        # stricter (lower accepts fewer)
CONFIRM_FRAMES = 3      # require this many identical recognitions in a row
RECOGNIZE_EVERY_N_FRAMES = 2  # speed up: only compute embeddings every N frames

print("Loading known students...")
known = load_known()
if not known:
    print("No enrolled students found. Run enroll_simple.py first.")
    exit()
print("Known:", [k[1] for k in known])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

frame_idx = 0
tracks = {}   # track_id -> {bbox, last_seen, confirm_label, confirm_count, last_identity}

next_track_id = 0

def assign_tracks(detected_boxes, tracks, max_dist_sq=200**2):
    """
    Simple track assignment by center distance.
    Return list of (track_id, box) for boxes in this frame.
    """
    global next_track_id
    assignments = []
    used = set()
    for box in detected_boxes:
        x1,y1,x2,y2 = [int(b) for b in box]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        best_tid = None; best_d = None
        for tid, t in tracks.items():
            bx1,by1,bx2,by2 = t['bbox']; bcx,bcy = (bx1+bx2)//2,(by1+by2)//2
            d = (bcx-cx)**2 + (bcy-cy)**2
            if best_d is None or d < best_d:
                best_d = d; best_tid = tid
        if best_tid is None or best_d > max_dist_sq:
            tid = next_track_id; next_track_id += 1
            tracks[tid] = {'bbox':(x1,y1,x2,y2), 'last_seen':time.time(),
                           'confirm_label': None, 'confirm_count': 0, 'last_identity': None}
            assignments.append((tid, (x1,y1,x2,y2)))
        else:
            tid = best_tid
            tracks[tid]['bbox'] = (x1,y1,x2,y2)
            tracks[tid]['last_seen'] = time.time()
            assignments.append((tid, (x1,y1,x2,y2)))
    return assignments

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read frame.")
        break
    frame_idx += 1
    boxes, _ = mtcnn.detect(frame)
    now = time.time()
    # expire old tracks (not seen for > 2s)
    to_delete = [tid for tid,t in tracks.items() if now - t['last_seen'] > 2.0]
    for tid in to_delete:
        del tracks[tid]

    display = frame.copy()

    if boxes is not None:
        assigned = assign_tracks(boxes, tracks)
        for tid, bbox in assigned:
            x1,y1,x2,y2 = bbox
            label = "..."
            # Only compute embeddings every N frames per track to save CPU
            if frame_idx % RECOGNIZE_EVERY_N_FRAMES == 0:
                try:
                    face_tensor = mtcnn.extract(frame, [bbox], None)[0]
                except Exception:
                    face_tensor = None
                if face_tensor is not None:
                    with torch.no_grad():
                        emb = resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
                    embn = emb / np.linalg.norm(emb)
                    dists = [cosine_distance(embn, k[2]) for k in known]
                    best_idx = int(np.argmin(dists))
                    best_dist = dists[best_idx]
                    if best_dist < THRESHOLD:
                        sid, name = known[best_idx][0], known[best_idx][1]
                        predicted = (sid, name)
                    else:
                        predicted = None
                else:
                    predicted = None

                # confirmation logic
                t = tracks[tid]
                if predicted is None:
                    # reset confirmation if unknown
                    t['confirm_label'] = None
                    t['confirm_count'] = 0
                else:
                    sid, name = predicted
                    if t['confirm_label'] == sid:
                        t['confirm_count'] += 1
                    else:
                        t['confirm_label'] = sid
                        t['confirm_count'] = 1

                    if t['confirm_count'] >= CONFIRM_FRAMES:
                        # confirmation reached
                        if t['last_identity'] != sid:
                            # new confirmed identity -> register attendance
                            add_attendance(sid)
                            t['last_identity'] = sid
                        label = f"{name} ({best_dist:.2f})"
                    else:
                        label = f"{name} ?"
            else:
                # in-between frames show last known identity if confirmed
                t = tracks[tid]
                if t['last_identity'] is not None:
                    # find name from id
                    nid = t['last_identity']
                    name = next((k[1] for k in known if k[0]==nid), None)
                    label = f"{name}"
                else:
                    label = "..."
            # draw
            cv2.rectangle(display, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.putText(display, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Attendance (safe)", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
