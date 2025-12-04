# Face Recognition Attendance System

A simple and effective face recognition system for tracking attendance.

## Features
- **Enrollment**: Capture face samples and store embeddings in a SQLite database.
- **Recognition**: Real-time face recognition with "liveness" checks (consecutive frame confirmation) to prevent false positives.
- **Attendance Tracking**: Automatically records attendance once per day for recognized individuals.

## Prerequisites

Ensure you have Python installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1.  Initialize the database:
    ```bash
    python db_setup.py
    ```
    This will create `attendance.db` with the necessary tables (`students`, `attendance`).

## Usage

### 1. Enroll a New Student

Run the enrollment script to add a new person to the database.

```bash
python enroll_simple.py
```
- Enter the student's name when prompted.
- The camera will open. Press **'c'** to capture a photo.
- Capture at least 6 photos for better accuracy.
- Press **'q'** when finished.

### 2. Start Recognition

Run the recognition script to start tracking attendance.

```bash
python recognize_safe.py
```
- The system will detect faces and attempt to match them against the database.
- If a face matches a known student for **3 consecutive frames**, attendance is recorded.
- Press **'q'** to quit.

## Files
- `enroll_simple.py`: Script to enroll new users.
- `recognize_safe.py`: Main script for recognition and attendance.
- `db_setup.py`: Utility to create the database schema.
- `attendance.db`: SQLite database storing user embeddings and attendance records.