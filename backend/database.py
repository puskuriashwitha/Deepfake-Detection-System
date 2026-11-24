import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parents[1] / "results" / "history.db"

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                duration_seconds REAL,
                sampled_frames INTEGER,
                faces_detected INTEGER,
                status TEXT NOT NULL
            )
        """)

def add_analysis(filename, duration, sampled, faces, status):
    with connect() as con:
        con.execute(
            "INSERT INTO analyses(filename, created_at, duration_seconds, sampled_frames, faces_detected, status) VALUES (?, ?, ?, ?, ?, ?)",
            (filename, datetime.now(timezone.utc).isoformat(), duration, sampled, faces, status)
        )

def get_history(limit=50):
    with connect() as con:
        rows = con.execute(
            "SELECT id, filename, created_at, duration_seconds, sampled_frames, faces_detected, status FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    keys = ["id", "filename", "created_at", "duration_seconds", "sampled_frames", "faces_detected", "status"]
    return [dict(zip(keys, row)) for row in rows]
