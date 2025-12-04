# db_setup.py
import sqlite3, pickle, os

DB = "attendance.db"

def create_tables():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        embedding BLOB NOT NULL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        date_iso TEXT,
        first_seen_ts REAL,
        last_seen_ts REAL,
        count INTEGER,
        FOREIGN KEY(student_id) REFERENCES students(id)
    )""")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_tables()
    print("Database and tables ready:", os.path.abspath(DB))
