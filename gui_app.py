import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import sqlite3
import pickle
import numpy as np
import time
import datetime
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import threading

# Configuration
DB = "attendance.db"
DEVICE = 'cpu'

class FaceRecoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")

        # Initialize AI models (lazy load or thread could be better, but simple here)
        self.status_var = tk.StringVar()
        self.status_var.set("Loading models... Please wait.")
        self.status_label = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Force UI update to show loading
        self.root.update()

        try:
            self.mtcnn = MTCNN(keep_all=True, device=DEVICE)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
            self.status_var.set("Models loaded. Ready.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            self.root.destroy()
            return

        # Container for frames
        self.container = tk.Frame(root)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MainMenu, EnrollPage, AttendancePage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenu")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

    def get_models(self):
        return self.mtcnn, self.resnet

class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Face Recognition System", font=("Helvetica", 24))
        label.pack(side="top", fill="x", pady=50)

        btn_enroll = tk.Button(self, text="Enroll New Student", font=("Helvetica", 16),
                               command=lambda: controller.show_frame("EnrollPage"), height=2, width=20)
        btn_enroll.pack(pady=10)

        btn_attend = tk.Button(self, text="Start Attendance", font=("Helvetica", 16),
                               command=lambda: controller.show_frame("AttendancePage"), height=2, width=20)
        btn_attend.pack(pady=10)

        btn_exit = tk.Button(self, text="Exit", font=("Helvetica", 16),
                             command=controller.root.quit, height=2, width=20)
        btn_exit.pack(pady=10)

class EnrollPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.cap = None
        self.samples = []
        self.is_running = False

        # UI Layout
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        tk.Button(top_frame, text="< Back", command=self.go_back).pack(side=tk.LEFT)
        tk.Label(top_frame, text="Enrollment", font=("Helvetica", 18)).pack(side=tk.LEFT, padx=20)

        self.name_var = tk.StringVar()
        tk.Label(top_frame, text="Name:").pack(side=tk.LEFT)
        self.entry_name = tk.Entry(top_frame, textvariable=self.name_var)
        self.entry_name.pack(side=tk.LEFT, padx=5)

        self.lbl_video = tk.Label(self)
        self.lbl_video.pack(expand=True)

        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        self.btn_capture = tk.Button(btn_frame, text="Capture Photo (0/6)", command=self.capture_sample, state=tk.DISABLED)
        self.btn_capture.pack(side=tk.LEFT, padx=20, expand=True)

        self.btn_save = tk.Button(btn_frame, text="Save & Finish", command=self.save_enrollment, state=tk.DISABLED)
        self.btn_save.pack(side=tk.RIGHT, padx=20, expand=True)

    def on_show(self):
        self.samples = []
        self.name_var.set("")
        self.update_buttons()
        self.start_camera()

    def go_back(self):
        self.stop_camera()
        self.controller.show_frame("MainMenu")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.update_video()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.lbl_video.configure(image='')

    def update_video(self):
        if not self.is_running:
            return
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # Convert to RGB for Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)
        
        self.after(20, self.update_video)

    def update_buttons(self):
        count = len(self.samples)
        self.btn_capture.config(text=f"Capture Photo ({count}/6)")
        if self.name_var.get().strip():
            self.btn_capture.config(state=tk.NORMAL)
        else:
            self.btn_capture.config(state=tk.DISABLED)
        
        if count >= 6:
            self.btn_save.config(state=tk.NORMAL)
        else:
            self.btn_save.config(state=tk.DISABLED)

    def capture_sample(self):
        if hasattr(self, 'current_frame'):
            self.samples.append(self.current_frame.copy())
            self.update_buttons()

    def save_enrollment(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter a name.")
            return
        if not self.samples:
            return

        self.controller.status_var.set("Processing enrollment...")
        self.root.update()

        mtcnn, resnet = self.controller.get_models()
        tensors = []
        for img in self.samples:
            try:
                # MTCNN expects RGB, but we have BGR from cv2
                # Actually facenet_pytorch MTCNN handles PIL or numpy. 
                # If numpy, it assumes RGB if we don't say otherwise? 
                # Standard is to pass RGB.
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face = mtcnn(img_rgb)
                if face is not None:
                    tensors.append(face)
            except Exception:
                pass

        if not tensors:
            messagebox.showerror("Error", "No faces detected in samples.")
            self.samples = []
            self.update_buttons()
            self.controller.status_var.set("Ready.")
            return

        stack = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            embeddings = resnet(stack).cpu().numpy()
        mean_emb = embeddings.mean(axis=0)

        try:
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute("INSERT INTO students (name, embedding) VALUES (?, ?)", (name, pickle.dumps(mean_emb)))
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", f"Enrolled {name} successfully!")
            self.go_back()
        except Exception as e:
            messagebox.showerror("Database Error", str(e))
        
        self.controller.status_var.set("Ready.")

class AttendancePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.cap = None
        self.is_running = False
        self.known = []
        self.tracks = {}
        self.frame_idx = 0
        self.next_track_id = 0

        # UI
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        tk.Button(top_frame, text="< Stop & Back", command=self.go_back).pack(side=tk.LEFT)
        tk.Label(top_frame, text="Attendance Mode", font=("Helvetica", 18)).pack(side=tk.LEFT, padx=20)

        self.lbl_video = tk.Label(self)
        self.lbl_video.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.log_text = tk.Text(self, width=30, state=tk.DISABLED)
        self.log_text.pack(side=tk.RIGHT, fill=tk.Y)

    def on_show(self):
        self.load_known()
        self.tracks = {}
        self.frame_idx = 0
        self.next_track_id = 0
        self.start_camera()

    def go_back(self):
        self.stop_camera()
        self.controller.show_frame("MainMenu")

    def load_known(self):
        try:
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute("SELECT id, name, embedding FROM students")
            rows = cur.fetchall()
            conn.close()
            self.known = []
            for sid, name, emb_blob in rows:
                emb = pickle.loads(emb_blob)
                emb = emb / np.linalg.norm(emb)
                self.known.append((sid, name, emb))
            self.log("Loaded known students.")
        except Exception as e:
            self.log(f"DB Error: {e}")

    def log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.process_loop()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.lbl_video.configure(image='')

    def process_loop(self):
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.process_loop)
            return

        # Recognition Logic (simplified from recognize_safe.py)
        mtcnn, resnet = self.controller.get_models()
        self.frame_idx += 1
        
        # Detect
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)
        
        display = frame.copy()
        
        if boxes is not None:
            # Simple tracking assignment would go here, but for GUI responsiveness 
            # let's keep it slightly simpler: just detect & recognize every few frames
            # or use the same logic if performance allows.
            
            # For simplicity in this GUI version, we'll do direct recognition on boxes
            # without the complex tracking object to keep code size manageable for the artifact.
            # If tracking is strictly needed, we can copy it. Let's do direct match for now.
            
            for box in boxes:
                x1,y1,x2,y2 = [int(b) for b in box]
                cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
                
                # Recognize every 3rd frame to save CPU
                if self.frame_idx % 3 == 0:
                    try:
                        # Extract
                        face_tensor = mtcnn.extract(img_rgb, [box], None)[0]
                        if face_tensor is not None:
                            emb = resnet(face_tensor.unsqueeze(0).to(DEVICE)).detach().cpu().numpy()[0]
                            embn = emb / np.linalg.norm(emb)
                            
                            # Match
                            best_name = "Unknown"
                            best_dist = 1.0
                            best_sid = None
                            
                            for sid, name, k_emb in self.known:
                                dist = 1 - np.dot(embn, k_emb)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_name = name
                                    best_sid = sid
                            
                            if best_dist < 0.45: # Threshold
                                label = f"{best_name} ({best_dist:.2f})"
                                self.record_attendance(best_sid, best_name)
                            else:
                                label = f"Unknown ({best_dist:.2f})"
                                
                            cv2.putText(display, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    except Exception:
                        pass

        # Update UI
        cv2image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lbl_video.imgtk = imgtk
        self.lbl_video.configure(image=imgtk)

        self.after(10, self.process_loop)

    def record_attendance(self, sid, name):
        # Check if already recorded today
        date_iso = datetime.date.today().isoformat()
        try:
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute("SELECT count FROM attendance WHERE student_id=? AND date_iso=?", (sid, date_iso))
            row = cur.fetchone()
            
            now = time.time()
            if row:
                # Update last seen
                # Don't spam log
                pass
            else:
                # Insert
                cur.execute("INSERT INTO attendance (student_id, date_iso, first_seen_ts, last_seen_ts, count) VALUES (?,?,?,?,?)",
                            (sid, date_iso, now, now, 1))
                conn.commit()
                self.log(f"MARKED: {name} at {datetime.datetime.now().strftime('%H:%M:%S')}")
            conn.close()
        except Exception as e:
            print(e)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecoApp(root)
    root.mainloop()
