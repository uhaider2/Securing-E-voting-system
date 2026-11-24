import sys
import asyncio

# --- Fix for Windows: aiortc/streamlit-webrtc raises noisy WinError 10054
# ConnectionResetError tracebacks on the default ProactorEventLoop when a
# WebRTC peer connection is torn down. SelectorEventLoop doesn't have this
# issue. This must run before streamlit_webrtc (or any asyncio usage) loads.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import cv2
st.write("OpenCV Version:", cv2.__version__)
st.write("CascadeClassifier available:", hasattr(cv2, "CascadeClassifier"))
import os
import math
import html
import ipaddress
import threading
import string
import smtplib
import hashlib
import hmac
import secrets
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ==================== CONFIGURATION ====================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evoting.db")

ORG_NAME = "University Student Council"

# Allowed voting location (latitude, longitude) and radius in meters.
ALLOWED_CENTER = (30.1872, 71.4750)   # TESTING: Delhi Gate, Multan (old walled city, Alang Road)
ALLOWED_RADIUS_METERS = 1000

# CNN face-embedding match threshold. Voter/candidate templates are 512-d
# InsightFace ArcFace (buffalo_l) embeddings, L2-normalized, compared with
# cosine similarity. Higher similarity = better match (1.0 = identical).
FACE_MATCH_MIN_SIMILARITY = 0.38

# YOLOv8-face (Ultralytics) confidence threshold for treating a detection as
# a real face.
YOLO_FACE_CONF = 0.40

# Number of consecutive frames a face must be steadily detected before the
# registration cameras auto-capture (higher = more stable, but slower).
STEADY_FACE_FRAMES_REQUIRED = 8

# ---- SMTP (real email) configuration ----
# Set these as environment variables before launching the app to send real
# emails (OTP codes, candidacy approval/rejection notices). If left blank,
# the app falls back to logging the email content to the terminal.
SMTP_HOST = os.environ.get("EVOTING_SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("EVOTING_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("EVOTING_SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("EVOTING_SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("EVOTING_SMTP_FROM", SMTP_USER)
OTP_VALID_MINUTES = 10

# TESTING ONLY: when SMTP is not configured, show the OTP on screen so the
# candidate registration flow can be tested. Set to False before real deployment.
DEV_SHOW_OTP_WHEN_NO_SMTP = True

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
EYE_GLASSES_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"

LOCAL_CASCADE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades")

CASCADE_DOWNLOAD_URLS = {
    "haarcascade_frontalface_default.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
    "haarcascade_eye_tree_eyeglasses.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
}

def resolve_cascade_path(filename):
    """
    Find a working path for a Haar cascade XML file: bundled opencv data,
    then a local ./cascades/ cache, then download fresh from OpenCV's repo.
    """
    bundled = os.path.join(cv2.data.haarcascades, filename)
    if os.path.isfile(bundled) and os.path.getsize(bundled) > 0:
        return bundled

    local_path = os.path.join(LOCAL_CASCADE_DIR, filename)
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    url = CASCADE_DOWNLOAD_URLS.get(filename)
    if url:
        try:
            os.makedirs(LOCAL_CASCADE_DIR, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(url, local_path)
            if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
                return local_path
        except Exception as e:
            print(f"[cascade download] Could not fetch '{filename}': {e}")
    return None

_face_path = resolve_cascade_path("haarcascade_frontalface_default.xml")
_eye_path = resolve_cascade_path("haarcascade_eye.xml")
_eye_glasses_path = resolve_cascade_path("haarcascade_eye_tree_eyeglasses.xml")

_face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(_face_path)

if face_cascade.empty():
    st.error("Face detection model could not be loaded!")
    st.stop()
eye_cascade = cv2.CascadeClassifier(_eye_path) if _eye_path else cv2.CascadeClassifier()
eye_glasses_cascade = cv2.CascadeClassifier(_eye_glasses_path) if _eye_glasses_path else cv2.CascadeClassifier()

CASCADES_OK = not (face_cascade.empty() or eye_cascade.empty() or eye_glasses_cascade.empty())

FACE_SIZE = (200, 200)

# ---- CNN face detector: YOLOv8-face (Ultralytics) ----
# Used as the primary face detector for both live auto-capture and biometric
# registration; falls back to the Haar cascade above only if these weights
# could not be downloaded (e.g. no internet on first run).
YOLO_FACE_WEIGHTS_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
YOLO_FACE_WEIGHTS_PATH = os.path.join(LOCAL_CASCADE_DIR, "yolov8n-face.pt")

def resolve_yolo_face_weights():
    if os.path.isfile(YOLO_FACE_WEIGHTS_PATH) and os.path.getsize(YOLO_FACE_WEIGHTS_PATH) > 1_000_000:
        return YOLO_FACE_WEIGHTS_PATH
    try:
        os.makedirs(LOCAL_CASCADE_DIR, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(YOLO_FACE_WEIGHTS_URL, YOLO_FACE_WEIGHTS_PATH)
        if os.path.isfile(YOLO_FACE_WEIGHTS_PATH) and os.path.getsize(YOLO_FACE_WEIGHTS_PATH) > 1_000_000:
            return YOLO_FACE_WEIGHTS_PATH
    except Exception as e:
        print(f"[YOLO weights download] Could not fetch yolov8n-face.pt: {e}")
    return None

@st.cache_resource(show_spinner="Loading YOLOv8-face CNN model (one-time per app restart)...")
def load_yolo_face_model():
    """
    Cached so this heavy model is loaded ONCE per app process instead of on
    every single click/interaction (Streamlit re-runs the whole script on
    every widget interaction - without caching this made the UI feel like it
    needed a "double click" because the first click was just reloading the
    model).
    """
    weights = resolve_yolo_face_weights()
    if not weights:
        return None
    try:
        return YOLO(weights)
    except Exception as e:
        print(f"[YOLO load] Could not load yolov8n-face.pt: {e}")
        return None

yolo_face_model = load_yolo_face_model()
YOLO_OK = yolo_face_model is not None

# ---- CNN face embedding/recognition: InsightFace ArcFace (buffalo_l) ----
# On first run this downloads its model pack automatically (needs internet
# once) to ~/.insightface/models/buffalo_l/. Produces 512-d, L2-normalized
# identity embeddings used for voter/candidate matching (FR-4 / FR-6).
@st.cache_resource(show_spinner="Loading InsightFace ArcFace CNN model (one-time per app restart)...")
def load_insightface_app():
    """Cached for the same reason as load_yolo_face_model() above."""
    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        return app
    except Exception as e:
        print(f"[InsightFace load] Could not load ArcFace (buffalo_l) model: {e}")
        return None

_insightface_app = load_insightface_app()
INSIGHTFACE_OK = _insightface_app is not None


# ==================== DATABASE SETUP ====================

def get_conn():
    return sqlite3.connect(DB_PATH)

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS department (
        department_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_datetime TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS superadmin (
        superadmin_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT,
        created_datetime TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS dean (
        dean_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT,
        created_datetime TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS hod (
        hod_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT,
        department_id INTEGER,
        created_datetime TEXT
    )""")

    # Kept for backward compatibility with older databases. No longer used
    # directly by the UI (replaced by superadmin / dean / hod), but existing
    # rows are migrated into the 'dean' table on first run.
    c.execute("""CREATE TABLE IF NOT EXISTS admin (
        admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT,
        created_datetime TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS candidate (
        candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        roll_number TEXT,
        party_name TEXT,
        status TEXT DEFAULT 'pending',
        registered_datetime TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS voter (
        voter_id INTEGER PRIMARY KEY AUTOINCREMENT,
        cnic TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        roll_number TEXT,
        voting_status INTEGER DEFAULT 0,
        face_image BLOB,
        registered_datetime TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS vote (
        vote_id INTEGER PRIMARY KEY AUTOINCREMENT,
        voter_id INTEGER,
        candidate_id INTEGER,
        vote_timestamp TEXT,
        prev_hash TEXT,
        record_hash TEXT,
        receipt_code TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS voter_authentication_log (
        auth_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        voter_id INTEGER,
        datetime TEXT,
        auth_method TEXT,
        auth_status TEXT,
        face_match_confidence REAL,
        liveness_status TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS gps_radius_check_log (
        gps_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        auth_log_id INTEGER,
        captured_lat REAL,
        captured_lon REAL,
        distance_to_boundary REAL,
        within_radius INTEGER
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS society (
        society_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        department_id INTEGER,
        created_datetime TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS notice (
        notice_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        body TEXT,
        posted_by TEXT,
        department_id INTEGER,
        created_datetime TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS email_otp (
        email TEXT PRIMARY KEY,
        otp_code TEXT,
        expiry_datetime TEXT,
        verified INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS login_lockout (
        role_username TEXT PRIMARY KEY,
        fail_count INTEGER DEFAULT 0,
        locked_until TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS branding (
        setting_key TEXT PRIMARY KEY,
        setting_value TEXT
    )""")
    conn.commit()
    conn.close()

def _add_column_if_missing(c, table, column, coltype):
    c.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in c.fetchall()]
    if column not in cols:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

# ==================== SECURITY: PASSWORD HASHING ====================
# PBKDF2-HMAC-SHA256 with a per-password random salt (100k iterations).
# Uses only the standard library (hashlib/secrets) - no extra dependency.

PBKDF2_ITERATIONS = 100_000

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"),
                                  PBKDF2_ITERATIONS).hex()
    return f"{salt}${digest}"

def verify_password(password, stored):
    """Returns (matches, needs_rehash). needs_rehash=True means the stored value
    was legacy plain-text and should be re-saved as a hash after a successful login."""
    if not stored:
        return False, False
    if "$" not in stored:
        return password == stored, (password == stored)
    salt, digest = stored.split("$", 1)
    check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"),
                                 PBKDF2_ITERATIONS).hex()
    return hmac.compare_digest(check, digest), False

# ==================== SECURITY: LOGIN LOCKOUT (brute-force protection) ====================

LOCKOUT_FAIL_THRESHOLD = 5
LOCKOUT_SECONDS = 60

def check_lockout(role, username):
    """Returns (is_locked, seconds_remaining)."""
    key = f"{role}:{(username or '').strip().lower()}"
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT locked_until FROM login_lockout WHERE role_username=?", (key,))
    row = c.fetchone()
    conn.close()
    if not row or not row[0]:
        return False, 0
    remaining = (datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") - datetime.now()).total_seconds()
    return (remaining > 0), max(0, int(remaining))

def record_login_result(role, username, success):
    key = f"{role}:{(username or '').strip().lower()}"
    conn = get_conn(); c = conn.cursor()
    if success:
        c.execute("DELETE FROM login_lockout WHERE role_username=?", (key,))
        conn.commit(); conn.close()
        return
    c.execute("SELECT fail_count FROM login_lockout WHERE role_username=?", (key,))
    row = c.fetchone()
    fail_count = (row[0] if row else 0) + 1
    locked_until = None
    if fail_count >= LOCKOUT_FAIL_THRESHOLD:
        locked_until = (datetime.now() + timedelta(seconds=LOCKOUT_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")
        fail_count = 0  # reset counter once a lockout window is issued
    c.execute("""INSERT INTO login_lockout (role_username, fail_count, locked_until) VALUES (?,?,?)
                 ON CONFLICT(role_username) DO UPDATE SET fail_count=excluded.fail_count,
                 locked_until=excluded.locked_until""", (key, fail_count, locked_until))
    conn.commit(); conn.close()

# ==================== SECURITY: CNIC PROTECTION (NFR-10 privacy) ====================
# CNIC is never stored in the clear. A keyed HMAC (deterministic, so it can
# still be used for uniqueness/lookup) is stored in `cnic_hash`, and only a
# masked form (e.g. *********1234) is kept in the visible `cnic` column.

SECRET_KEY_PATH = os.path.join(LOCAL_CASCADE_DIR, "secret.key")

def _get_or_create_secret_key():
    try:
        if os.path.isfile(SECRET_KEY_PATH):
            with open(SECRET_KEY_PATH, "r") as f:
                key = f.read().strip()
                if key:
                    return key
        os.makedirs(LOCAL_CASCADE_DIR, exist_ok=True)
        key = secrets.token_hex(32)
        with open(SECRET_KEY_PATH, "w") as f:
            f.write(key)
        return key
    except Exception as e:
        print(f"[secret key] Could not persist secret key, using a session-only key: {e}")
        return secrets.token_hex(32)

CNIC_SECRET_KEY = _get_or_create_secret_key()

def hash_cnic(digits):
    return hmac.new(CNIC_SECRET_KEY.encode("utf-8"), digits.encode("utf-8"), hashlib.sha256).hexdigest()

def mask_cnic(digits):
    return "*" * (len(digits) - 4) + digits[-4:]

# ==================== SECURITY: TAMPER-EVIDENT VOTE LEDGER ====================
# Every vote row is chained to the previous one (like a lightweight blockchain):
# record_hash = SHA256(voter_id | candidate_id | timestamp | prev_hash of the
# previous vote). Any edit/delete/reorder of a past vote breaks the chain from
# that point onward, which verify_vote_chain() below can detect and prove.

GENESIS_HASH = "0" * 64

def _compute_vote_hash(voter_id, candidate_id, vote_timestamp, prev_hash):
    payload = f"{voter_id}|{candidate_id}|{vote_timestamp}|{prev_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def generate_receipt_code():
    return secrets.token_hex(5).upper()

def migrate_db():
    """Adds columns/seed data for older databases created before these fields existed."""
    conn = get_conn()
    c = conn.cursor()

    for table in ("candidate", "voter"):
        _add_column_if_missing(c, table, "roll_number", "TEXT")
        _add_column_if_missing(c, table, "society_id", "INTEGER")
        _add_column_if_missing(c, table, "cnic_hash", "TEXT")
    _add_column_if_missing(c, "candidate", "cnic", "TEXT")
    _add_column_if_missing(c, "candidate", "email_verified", "INTEGER DEFAULT 0")
    _add_column_if_missing(c, "voter", "face_encoding", "BLOB")
    _add_column_if_missing(c, "society", "department_id", "INTEGER")
    _add_column_if_missing(c, "notice", "department_id", "INTEGER")
    _add_column_if_missing(c, "vote", "prev_hash", "TEXT")
    _add_column_if_missing(c, "vote", "record_hash", "TEXT")
    _add_column_if_missing(c, "vote", "receipt_code", "TEXT")
    # NEW: email address for each admin account, used for approval notifications
    for admin_table in ("superadmin", "dean", "hod"):
        _add_column_if_missing(c, admin_table, "email", "TEXT")
    conn.commit()

    # ---- Security migration: hash any un-hashed CNICs left over from an ----
    # ---- older version of this database (plaintext 13-digit values) ----
    for table in ("voter", "candidate"):
        c.execute(f"SELECT rowid, cnic FROM {table} WHERE cnic_hash IS NULL AND cnic IS NOT NULL")
        rows = c.fetchall()
        for rowid, cnic_val in rows:
            digits = "".join(ch for ch in str(cnic_val) if ch.isdigit())
            if len(digits) == 13:
                c.execute(f"UPDATE {table} SET cnic_hash=?, cnic=? WHERE rowid=?",
                          (hash_cnic(digits), mask_cnic(digits), rowid))
        if rows:
            conn.commit()

    # ---- Security migration: hash any legacy plain-text admin passwords ----
    for table in ("superadmin", "dean", "hod"):
        id_col = f"{table}_id"
        c.execute(f"SELECT {id_col}, password FROM {table}")
        for row_id, pw in c.fetchall():
            if pw and "$" not in pw:
                c.execute(f"UPDATE {table} SET password=? WHERE {id_col}=?", (hash_password(pw), row_id))
        conn.commit()

    # ---- Seed a default department so societies always have somewhere to go ----
    c.execute("SELECT COUNT(*) FROM department")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO department (name, created_datetime) VALUES (?,?)",
                  ("General", now_str()))
        conn.commit()
    c.execute("SELECT department_id FROM department ORDER BY department_id LIMIT 1")
    default_dept_id = c.fetchone()[0]
    c.execute("UPDATE society SET department_id=? WHERE department_id IS NULL", (default_dept_id,))
    conn.commit()

    # ---- Seed a default Super Admin (change this password after first login) ----
    c.execute("SELECT COUNT(*) FROM superadmin")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO superadmin (username, password, full_name, created_datetime) VALUES (?,?,?,?)",
                  ("superadmin", hash_password("super123"), "Default Super Admin", now_str()))
        conn.commit()

    # ---- Migrate any pre-existing 'admin' rows into 'dean' (institute-wide admin) ----
    c.execute("SELECT COUNT(*) FROM dean")
    if c.fetchone()[0] == 0:
        c.execute("SELECT username, password, full_name, created_datetime FROM admin")
        old_admin_rows = c.fetchall()
        if old_admin_rows:
            for username, password, full_name, created_datetime in old_admin_rows:
                try:
                    stored_pw = password if (password and "$" in password) else hash_password(password or "")
                    c.execute("INSERT INTO dean (username, password, full_name, created_datetime) VALUES (?,?,?,?)",
                              (username, stored_pw, full_name, created_datetime))
                except sqlite3.IntegrityError:
                    pass
        else:
            c.execute("INSERT INTO dean (username, password, full_name, created_datetime) VALUES (?,?,?,?)",
                      ("dean", hash_password("dean123"), "Default Dean", now_str()))
        conn.commit()

    conn.close()

init_db()
migrate_db()

# ==================== BRANDING LOGIC ====================

import base64

DEFAULT_PRIMARY_COLOR = "#2563EB"
DEFAULT_PRIMARY_DARK = "#1E3A8A"

def get_branding(key, default=None):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT setting_value FROM branding WHERE setting_key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default

def set_branding(key, value):
    conn = get_conn(); c = conn.cursor()
    c.execute("""INSERT INTO branding (setting_key, setting_value) VALUES (?,?)
                 ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value""", (key, value))
    conn.commit(); conn.close()

def set_logo_from_upload(uploaded_file):
    if uploaded_file is None:
        return False, "No file selected."
    try:
        data = uploaded_file.getvalue()
        if len(data) > 2_000_000:
            return False, "Logo file too large (max 2MB)."
        b64 = base64.b64encode(data).decode("ascii")
        mime = uploaded_file.type or "image/png"
        set_branding("logo_data_uri", f"data:{mime};base64,{b64}")
        return True, "Logo updated successfully."
    except Exception as e:
        return False, f"Could not process logo: {e}"

def get_logo_data_uri():
    return get_branding("logo_data_uri")

def clear_logo():
    conn = get_conn(); c = conn.cursor()
    c.execute("DELETE FROM branding WHERE setting_key='logo_data_uri'")
    conn.commit(); conn.close()

def get_org_name():
    return get_branding("org_name") or ORG_NAME

def set_org_name(name):
    if name and name.strip():
        set_branding("org_name", name.strip())

def get_primary_color():
    return get_branding("primary_color") or DEFAULT_PRIMARY_COLOR

def get_primary_dark():
    return get_branding("primary_dark") or DEFAULT_PRIMARY_DARK

def set_theme_colors(primary, dark):
    if primary:
        set_branding("primary_color", primary)
    if dark:
        set_branding("primary_dark", dark)


# ==================== HELPER FUNCTIONS ====================

def is_valid_cnic(cnic):
    digits = "".join(ch for ch in str(cnic) if ch.isdigit())
    if len(digits) == 13:
        return True, digits
    return False, None

def camera_file_to_array(camera_file):
    if camera_file is None:
        return None
    image = Image.open(camera_file).convert("RGB")
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _detect_faces(image_bgr, min_size=(60, 60)):
    """
    CNN-based face detection using YOLOv8-face (Ultralytics). Falls back to
    the Haar cascade only if the YOLO weights could not be loaded (e.g. no
    internet was available on first run to fetch the model). Always returns
    a list of (x, y, w, h) boxes in pixel coordinates.
    """
    if YOLO_OK:
        try:
            results = yolo_face_model.predict(image_bgr, conf=YOLO_FACE_CONF, verbose=False)
            boxes = []
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    w, h = x2 - x1, y2 - y1
                    if w >= min_size[0] and h >= min_size[1]:
                        boxes.append((int(x1), int(y1), int(w), int(h)))
            if boxes:
                return boxes
        except Exception as e:
            print(f"[YOLO face detect] error, falling back to Haar cascade: {e}")

    gray = cv2.equalizeHist(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
    attempts = [
        dict(scaleFactor=1.1, minNeighbors=5, minSize=min_size),
        dict(scaleFactor=1.05, minNeighbors=4, minSize=min_size),
        dict(scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)),
    ]
    for params in attempts:
        faces = face_cascade.detectMultiScale(gray, **params)
        if len(faces) > 0:
            return list(faces)
    return []

def extract_face_crop(image, size=FACE_SIZE):
    if image is None:
        return None, "No image captured."
    boxes = _detect_faces(image)
    if len(boxes) == 0:
        return None, "No face detected. Please retake the photo with better lighting, facing the camera directly."
    x, y, w, h = max(boxes, key=lambda f: f[2] * f[3])
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    x0, y0 = max(0, x), max(0, y)
    face = gray[y0:y0 + h, x0:x0 + w]
    if face.size == 0:
        return None, "Face crop was empty. Please retake the photo."
    face = cv2.resize(face, size)
    face = cv2.equalizeHist(face)
    return face, "Face detected."

def extract_face_encoding(image):
    """
    CNN-based facial biometric extraction (FR-4 / FR-6).
    YOLOv8-face (Ultralytics CNN) first confirms a real face is present, then
    InsightFace's ArcFace CNN (buffalo_l, ResNet-based) performs its own
    detection + 5-point landmark alignment + embedding to produce a 512-d,
    L2-normalized deep face embedding used for identity matching.
    Returns (embedding_ndarray_or_None, gray_face_crop_or_None, message).
    """
    if image is None:
        return None, None, "No image captured."
    boxes = _detect_faces(image)
    if len(boxes) == 0:
        return None, None, "No face detected. Please retake the photo with better lighting, facing the camera directly."
    if not INSIGHTFACE_OK:
        return None, None, ("Face recognition model is not available (InsightFace ArcFace failed to load - "
                             "it needs internet access on first run to download its model pack). "
                             "Please restart the app with an internet connection.")

    try:
        faces = _insightface_app.get(image)
    except Exception as e:
        return None, None, f"CNN face-embedding extraction failed: {e}"

    if not faces:
        return None, None, "Could not extract facial features (CNN embedding failed). Please retake with better lighting."

    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    encoding = np.asarray(best.normed_embedding, dtype=np.float32)

    x, y, w, h = max(boxes, key=lambda f: f[2] * f[3])
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    x0, y0 = max(0, x), max(0, y)
    face_crop = gray[y0:y0 + h, x0:x0 + w]
    if face_crop.size == 0:
        face_crop = gray
    face_crop = cv2.resize(face_crop, FACE_SIZE)
    face_crop = cv2.equalizeHist(face_crop)
    return encoding, face_crop, "Face detected and CNN (ArcFace) facial embedding extracted."

def face_to_blob(face_array):
    return face_array.astype(np.uint8).tobytes()

def blob_to_face(blob, size=FACE_SIZE):
    arr = np.frombuffer(blob, dtype=np.uint8)
    return arr.reshape(size)

def encoding_to_blob(encoding):
    return np.asarray(encoding, dtype=np.float32).tobytes()

def blob_to_encoding(blob):
    return np.frombuffer(blob, dtype=np.float32)

def generate_otp():
    # FIX: uses `secrets` (cryptographically secure) instead of `random`.
    return "".join(secrets.choice(string.digits) for _ in range(6))

def send_email(to_email, subject, body):
    """
    Sends a real email via SMTP when EVOTING_SMTP_* environment variables are
    configured. Falls back to logging the message to the terminal so the app
    still works out-of-the-box during development/testing.
    """
    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
        print(f"[EMAIL -> {to_email}] {subject}\n{body}\n")
        return True, ("(SMTP not configured - notification logged to terminal. "
                       "Set EVOTING_SMTP_HOST / EVOTING_SMTP_USER / EVOTING_SMTP_PASSWORD "
                       "environment variables to send real emails.)")
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM or SMTP_USER
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM or SMTP_USER, [to_email], msg.as_string())
        return True, "Email sent successfully."
    except Exception as e:
        print(f"[EMAIL -> {to_email}] {subject}\n{body}\n[SMTP ERROR] {e}\n")
        return False, f"(Email sending failed: {e}. Notification logged to terminal instead.)"

def send_candidate_otp(email):
    if not email or "@" not in email:
        return False, "Enter a valid email address first."
    email = email.strip()
    otp = generate_otp()
    expiry = (datetime.now() + timedelta(minutes=OTP_VALID_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn(); c = conn.cursor()
    c.execute("""INSERT INTO email_otp (email, otp_code, expiry_datetime, verified) VALUES (?,?,?,0)
                 ON CONFLICT(email) DO UPDATE SET otp_code=excluded.otp_code,
                 expiry_datetime=excluded.expiry_datetime, verified=0""", (email, otp, expiry))
    conn.commit(); conn.close()
    sent, email_msg = send_email(
        email, "Your Candidacy Email Verification Code",
        f"Your one-time verification code is: {otp}\nThis code expires in {OTP_VALID_MINUTES} minutes.")
    dev_hint = ""
    if DEV_SHOW_OTP_WHEN_NO_SMTP and not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
        dev_hint = f" [TEST MODE] Your code is: {otp}"
    return True, f"Verification code sent to {email}. {email_msg}{dev_hint}"

def verify_candidate_otp(email, code):
    if not email or not code:
        return False, "Enter your email and the verification code."
    email = email.strip()
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT otp_code, expiry_datetime FROM email_otp WHERE email=?", (email,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "Please request a verification code for this email first."
    otp_code, expiry_datetime = row
    if datetime.now() > datetime.strptime(expiry_datetime, "%Y-%m-%d %H:%M:%S"):
        conn.close()
        return False, "Verification code expired. Please request a new one."
    if code.strip() != otp_code:
        conn.close()
        return False, "Incorrect verification code."
    c.execute("UPDATE email_otp SET verified=1 WHERE email=?", (email,))
    conn.commit(); conn.close()
    return True, "Email verified successfully."

def is_email_verified(email):
    if not email:
        return False
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT verified FROM email_otp WHERE email=?", (email.strip(),))
    row = c.fetchone()
    conn.close()
    return bool(row and row[0] == 1)

# ==================== NOTIFICATION EMAILS (NEW) ====================
# 1) A candidate registers      -> email goes to ALL Super Admins + Deans (please approve).
# 2) A candidate gets approved  -> email goes to the HOD(s) of that candidate's department.
# Emails are only sent to accounts that have an email address saved.
# (If SMTP is not configured, the emails are printed in the terminal instead.)

def _valid_email(email):
    email = (email or "").strip()
    return bool(email) and "@" in email and "." in email.split("@")[-1]

def _emails_for(query, params=()):
    conn = get_conn(); c = conn.cursor()
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return sorted({r[0].strip() for r in rows if _valid_email(r[0])})

def _candidate_details(candidate_id):
    conn = get_conn(); c = conn.cursor()
    c.execute("""SELECT c.name, c.email, c.roll_number, c.party_name, s.department_id, d.name
                 FROM candidate c
                 LEFT JOIN society s ON s.society_id = c.society_id
                 LEFT JOIN department d ON d.department_id = s.department_id
                 WHERE c.candidate_id=?""", (int(candidate_id),))
    row = c.fetchone()
    conn.close()
    return row

def notify_admins_new_candidate(candidate_id):
    """New candidate registered -> ask Super Admin(s) and Dean(s) to approve."""
    details = _candidate_details(candidate_id)
    if not details:
        return "No candidate found to notify about."
    name, cand_email, roll, society, dept_id, dept_name = details
    recipients = sorted(set(_emails_for("SELECT email FROM superadmin") +
                            _emails_for("SELECT email FROM dean")))
    if not recipients:
        return "No Super Admin / Dean email is saved, so no approval request was sent."
    subject = f"Action needed: approve candidate {name}"
    body = (f"A new candidate has registered and is waiting for your approval.\n\n"
            f"Name: {name}\n"
            f"Roll number: {roll}\n"
            f"Email: {cand_email}\n"
            f"Society: {society}\n"
            f"Department: {dept_name or 'N/A'}\n\n"
            f"Please log in to the Admin Panel and approve or reject this candidate.\n\n"
            f"Regards,\nE-Voting System")
    for addr in recipients:
        send_email(addr, subject, body)
    return f"Approval request sent to {len(recipients)} admin(s)."

def notify_hod_candidate_approved(candidate_id):
    """Candidate approved -> tell the HOD(s) of that department."""
    details = _candidate_details(candidate_id)
    if not details:
        return "No candidate found to notify about."
    name, cand_email, roll, society, dept_id, dept_name = details
    if not dept_id:
        return "Candidate has no department, so no HOD was notified."
    recipients = _emails_for("SELECT email FROM hod WHERE department_id=?", (dept_id,))
    if not recipients:
        return f"No HOD email is saved for department '{dept_name}', so no HOD was notified."
    subject = f"Candidate selected: {name} ({society})"
    body = (f"This student has been approved and selected as a candidate.\n\n"
            f"Name: {name}\n"
            f"Roll number: {roll}\n"
            f"Society: {society}\n"
            f"Department: {dept_name}\n\n"
            f"Regards,\nE-Voting System")
    for addr in recipients:
        send_email(addr, subject, body)
    return f"HOD notified ({len(recipients)} email(s))."

def set_account_email(role, account_id, email):
    table_map = {"Super Admin": ("superadmin", "superadmin_id"),
                 "Dean": ("dean", "dean_id"),
                 "HOD": ("hod", "hod_id")}
    if role not in table_map:
        return False, "Unknown account type."
    if not _valid_email(email):
        return False, "Please enter a valid email address."
    table, id_col = table_map[role]
    conn = get_conn(); c = conn.cursor()
    c.execute(f"UPDATE {table} SET email=? WHERE {id_col}=?", (email.strip(), int(account_id)))
    conn.commit(); conn.close()
    return True, "Email saved."

# ==================== ANTI-SPOOFING (FIXED) ====================
# OLD PROBLEM: the old check divided the average of the outer FFT spectrum by
# the average of the whole spectrum. That ratio is ~0.7-0.9 for EVERY image,
# so with a threshold of 0.09 every real face was flagged as "screen-like".
#
# NEW METHOD: remove the natural 1/f falloff of the spectrum, then look for
# isolated sharp spikes. Screens (moire patterns) create such spikes; real
# faces do not.

SPOOF_MOIRE_PEAK_RATIO = 15.0   # CALIBRATE: see the "[spoof]" lines printed in the terminal
SPOOF_MIN_SHARPNESS = 15.0
SPOOF_MAX_SHARPNESS = 8000.0    # equalized webcam crops are often > 2500
SPOOF_MIN_COLOR_STD = 8.0
SPOOF_DEBUG = True              # prints the scores to the terminal; set False when finished testing

def _moire_peak_ratio(face_gray):
    n = 128
    g = cv2.resize(face_gray, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)  # avoids edge "cross" artifact
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g * win)))

    yy, xx = np.ogrid[:n, :n]
    r = np.sqrt((yy - n // 2) ** 2 + (xx - n // 2) ** 2)
    r_int = r.astype(int)

    # Divide by the average magnitude at each radius to remove the 1/f falloff
    radial_mean = np.bincount(r_int.ravel(), weights=mag.ravel()) / np.maximum(np.bincount(r_int.ravel()), 1)
    norm = mag / (radial_mean[r_int] + 1e-6)

    ring = (r > n * 0.15) & (r < n * 0.5)
    return float(norm[ring].max())

def detect_screen_or_photo_spoof(face_gray, face_color=None):
    reasons = []

    peak = _moire_peak_ratio(face_gray)
    sharpness = cv2.Laplacian(face_gray, cv2.CV_64F).var()
    color_std = None

    if peak > SPOOF_MOIRE_PEAK_RATIO:
        reasons.append("screen-like periodic pattern detected")

    if sharpness < SPOOF_MIN_SHARPNESS:
        reasons.append("image too blurry (possible low-quality photo)")
    elif sharpness > SPOOF_MAX_SHARPNESS:
        reasons.append("image unnaturally sharp/processed")

    if face_color is not None:
        color_std = face_color.reshape(-1, face_color.shape[-1]).std(axis=0).mean()
        if color_std < SPOOF_MIN_COLOR_STD:
            reasons.append("unnaturally flat color/texture")

    if SPOOF_DEBUG:
        print(f"[spoof] moire_peak={peak:.1f} (thr {SPOOF_MOIRE_PEAK_RATIO}) "
              f"sharp={sharpness:.0f} color_std={color_std}")

    if reasons:
        return False, "Possible spoofing attempt: " + "; ".join(reasons) + "."
    return True, "No spoofing signals detected."

# ==================== EYE / BLINK DETECTION ====================

def _detect_eyes(gray):
    attempts = [
        (eye_cascade, dict(scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))),
        (eye_cascade, dict(scaleFactor=1.1, minNeighbors=5, minSize=(18, 18))),
        (eye_glasses_cascade, dict(scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))),
    ]
    best = []
    for cascade, params in attempts:
        eyes = cascade.detectMultiScale(gray, **params)
        if len(eyes) > len(best):
            best = eyes
        if len(best) >= 2:
            return best
    return best

def _detect_eyes_in_face(gray):
    """
    FIX: search for eyes ONLY inside the upper part of the detected face
    (not the whole frame), so posters/glasses reflections/background objects
    can't be mistaken for eyes.
    Returns None if no face is visible (so "face left the frame" is NOT
    mistaken for a blink), otherwise the list of detected eyes.
    """
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return _detect_eyes(gray[y:y + int(h * 0.6), x:x + w])

def _eye_count(gray):
    eyes = _detect_eyes_in_face(gray)
    return 0 if eyes is None else len(eyes)

def liveness_blink_check(frame_open, frame_blink):
    if frame_open is None or frame_blink is None:
        return False, "Both frames are required for the liveness check."

    gray_open = cv2.equalizeHist(cv2.cvtColor(frame_open, cv2.COLOR_BGR2GRAY))
    gray_blink = cv2.equalizeHist(cv2.cvtColor(frame_blink, cv2.COLOR_BGR2GRAY))

    n_open = _eye_count(gray_open)
    n_blink = _eye_count(gray_blink)

    blink_ok = n_open >= 1 and n_blink < n_open
    if not blink_ok:
        return False, ("Liveness check failed. Please retake: keep eyes open and facing "
                        "the camera in Frame 1, then blink clearly in Frame 2. Make sure your "
                        "face is well-lit and not too far from the camera.")

    face_open, _ = extract_face_crop(frame_open)
    face_blink, _ = extract_face_crop(frame_blink)
    if face_open is None or face_blink is None:
        return False, "Could not isolate face for anti-spoofing analysis. Please retake both frames."

    ok1, msg1 = detect_screen_or_photo_spoof(face_open, frame_open)
    ok2, msg2 = detect_screen_or_photo_spoof(face_blink, frame_blink)

    if not (ok1 and ok2):
        detail = msg1 if not ok1 else msg2
        return False, f"Anti-spoofing check failed - {detail} Please use a live camera feed, not a photo or screen."

    return True, "Liveness check passed (blink detected, no spoofing signals)."

def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def gps_check(lat, lon):
    dist = haversine_distance_m(lat, lon, ALLOWED_CENTER[0], ALLOWED_CENTER[1])
    return dist <= ALLOWED_RADIUS_METERS, dist

# ==================== LIVE AUTO-CAPTURE ====================

class AutoCaptureProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_open = None
        self.frame_blink = None
        self.state = "waiting_open"

    def reset(self):
        """FIX: lets the Retake button restart the capture from scratch."""
        with self.lock:
            self.frame_open = None
            self.frame_blink = None
            self.state = "waiting_open"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            # FIX: eyes searched only inside the face; None = no face visible
            eyes = _detect_eyes_in_face(gray) if CASCADES_OK else None

            with self.lock:
                if self.state == "waiting_open" and eyes is not None and len(eyes) >= 1:
                    self.frame_open = img.copy()
                    self.state = "waiting_blink"
                elif self.state == "waiting_blink" and eyes is not None and len(eyes) < 1:
                    self.frame_blink = img.copy()
                    self.state = "done"
                state = self.state

            if not CASCADES_OK:
                label = "ERROR: face cascade files not loaded (see terminal)"
                color = (0, 0, 255)
            else:
                label = {"waiting_open": "Look at camera, eyes open...",
                          "waiting_blink": "Now blink...",
                          "done": "Captured!"}[state]
                color = (0, 255, 0)
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            import traceback
            print("AutoCaptureProcessor.recv() error:", e)
            traceback.print_exc()
            cv2.putText(img, f"ERROR: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


class SteadyFaceCaptureProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.captured_frame = None
        self.streak = 0
        self.state = "waiting"

    def reset(self):
        """FIX: lets the Retake button restart the capture from scratch."""
        with self.lock:
            self.captured_frame = None
            self.streak = 0
            self.state = "waiting"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        label, color = "", (0, 255, 0)
        try:
            if not CASCADES_OK:
                label = "ERROR: face cascade files not loaded (see terminal)"
                color = (0, 0, 255)
            else:
                with self.lock:
                    if self.state == "waiting":
                        faces = _detect_faces(img)
                        if len(faces) > 0:
                            self.streak += 1
                        else:
                            self.streak = 0

                        if self.streak >= STEADY_FACE_FRAMES_REQUIRED:
                            self.captured_frame = img.copy()
                            self.state = "done"

                    remaining = max(0, STEADY_FACE_FRAMES_REQUIRED - self.streak)
                    label = "Captured!" if self.state == "done" else f"Hold still... {remaining}"
        except Exception as e:
            import traceback
            print("SteadyFaceCaptureProcessor.recv() error:", e)
            traceback.print_exc()
            label = f"ERROR: {e}"
            color = (0, 0, 255)

        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def render_auto_face_capture(key_prefix):
    ready_key = f"{key_prefix}_auto_ready"
    frame_key = f"{key_prefix}_auto_frame"

    st.caption("Turn on the camera and look straight ahead - once your face is steadily detected, "
               "the photo will be captured automatically. Step back slightly from the camera (~50-70cm) "
               "so your whole face fits in the frame.")

def render_auto_face_capture(key_prefix):
    ready_key = f"{key_prefix}_auto_ready"
    frame_key = f"{key_prefix}_auto_frame"

    ctx = webrtc_streamer(
        key=f"{key_prefix}-auto-capture",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SteadyFaceCaptureProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302"
                    ]
                }
            ]
        },
        async_processing=True,
    )

    if ctx.video_processor:
        with ctx.video_processor.lock:
            if (
                ctx.video_processor.state == "done"
                and not st.session_state.get(ready_key)
            ):
                st.session_state[frame_key] = (
                    ctx.video_processor.captured_frame.copy()
                )
                st.session_state[ready_key] = True

    if st.session_state.get(ready_key):
        st.success("Photo automatically captured.")

        st.image(
            cv2.cvtColor(
                st.session_state[frame_key],
                cv2.COLOR_BGR2RGB
            ),
            caption="Captured photo",
            width="stretch"
        )

        if st.button("Retake", key=f"{key_prefix}_retake"):
            if ctx.video_processor:
                ctx.video_processor.reset()

            st.session_state.pop(ready_key, None)
            st.session_state.pop(frame_key, None)
            st.rerun()

        return st.session_state[frame_key]

    else:
        st.info(
            "Place your face in front of the camera, looking straight ahead..."
        )
        return None
        rtc_configuration={
    "iceServers": [
        {
            "urls": ["stun:stun.l.google.com:19302"]
        },
        {
            "urls": ["stun:stun1.l.google.com:19302"]
        }
    ],
},
# ==================== DEPARTMENT LOGIC ====================

def add_department(name):
    if not name or not name.strip():
        return False, "Please enter a department name."
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("SELECT department_id FROM department WHERE name=?", (name.strip(),))
        if c.fetchone():
            conn.close()
            return False, "This department already exists."
        c.execute("INSERT INTO department (name, created_datetime) VALUES (?,?)", (name.strip(), now_str()))
        conn.commit(); conn.close()
        return True, f"Department '{name}' added successfully."
    except Exception as e:
        return False, f"Database error: {e}"

def list_departments():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT department_id AS ID, name AS Name, created_datetime AS Created FROM department ORDER BY name", conn)
    conn.close()
    return df

def get_department_options():
    df = list_departments()
    return {row.Name: row.ID for row in df.itertuples()}

# ==================== SUPER ADMIN LOGIC ====================

def authenticate_superadmin(username, password):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT superadmin_id, full_name, password FROM superadmin WHERE username=?", (username.strip(),))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    superadmin_id, full_name, stored_pw = row
    ok, needs_rehash = verify_password(password, stored_pw)
    if not ok:
        conn.close()
        return None
    if needs_rehash:
        c.execute("UPDATE superadmin SET password=? WHERE superadmin_id=?", (hash_password(password), superadmin_id))
        conn.commit()
    conn.close()
    return (superadmin_id, full_name)

def add_superadmin(username, password, full_name, email=""):
    if not username or not password:
        return False, "Username and password are both required."
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("SELECT superadmin_id FROM superadmin WHERE username=?", (username.strip(),))
        if c.fetchone():
            conn.close()
            return False, "This username already exists."
        c.execute("INSERT INTO superadmin (username, password, full_name, email, created_datetime) VALUES (?,?,?,?,?)",
                  (username.strip(), hash_password(password), (full_name or "").strip(), (email or "").strip(), now_str()))
        conn.commit(); conn.close()
        return True, f"New Super Admin '{username}' added successfully."
    except Exception as e:
        return False, f"Database error: {e}"

def list_superadmins():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT superadmin_id AS ID, username AS Username, full_name AS Name, email AS Email, created_datetime AS Registered FROM superadmin", conn)
    conn.close()
    return df

# ==================== DEAN LOGIC (institute-wide admin) ====================

def authenticate_dean(username, password):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT dean_id, full_name, password FROM dean WHERE username=?", (username.strip(),))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    dean_id, full_name, stored_pw = row
    ok, needs_rehash = verify_password(password, stored_pw)
    if not ok:
        conn.close()
        return None
    if needs_rehash:
        c.execute("UPDATE dean SET password=? WHERE dean_id=?", (hash_password(password), dean_id))
        conn.commit()
    conn.close()
    return (dean_id, full_name)

def add_dean(username, password, full_name, email=""):
    if not username or not password:
        return False, "Username and password are both required."
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("SELECT dean_id FROM dean WHERE username=?", (username.strip(),))
        if c.fetchone():
            conn.close()
            return False, "This username already exists."
        c.execute("INSERT INTO dean (username, password, full_name, email, created_datetime) VALUES (?,?,?,?,?)",
                  (username.strip(), hash_password(password), (full_name or "").strip(), (email or "").strip(), now_str()))
        conn.commit(); conn.close()
        return True, f"New Dean '{username}' added successfully."
    except Exception as e:
        return False, f"Database error: {e}"

def list_deans():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT dean_id AS ID, username AS Username, full_name AS Name, email AS Email, created_datetime AS Registered FROM dean", conn)
    conn.close()
    return df

def delete_dean(dean_id):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT username FROM dean WHERE dean_id=?", (int(dean_id),))
    row = c.fetchone()
    if not row:
        conn.close()
        return "Dean not found."
    c.execute("DELETE FROM dean WHERE dean_id=?", (int(dean_id),))
    conn.commit(); conn.close()
    return f"Dean '{row[0]}' removed."

# ==================== HOD LOGIC (department-scoped admin) ====================

def authenticate_hod(username, password):
    conn = get_conn(); c = conn.cursor()
    c.execute("""SELECT h.hod_id, h.full_name, h.department_id, d.name, h.password
                 FROM hod h LEFT JOIN department d ON d.department_id = h.department_id
                 WHERE h.username=?""", (username.strip(),))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    hod_id, full_name, department_id, dept_name, stored_pw = row
    ok, needs_rehash = verify_password(password, stored_pw)
    if not ok:
        conn.close()
        return None
    if needs_rehash:
        c.execute("UPDATE hod SET password=? WHERE hod_id=?", (hash_password(password), hod_id))
        conn.commit()
    conn.close()
    return (hod_id, full_name, department_id, dept_name)

def add_hod(username, password, full_name, department_id, email=""):
    if not username or not password:
        return False, "Username and password are both required."
    if not department_id:
        return False, "Please select a department for this HOD."
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("SELECT hod_id FROM hod WHERE username=?", (username.strip(),))
        if c.fetchone():
            conn.close()
            return False, "This username already exists."
        c.execute("INSERT INTO hod (username, password, full_name, email, department_id, created_datetime) VALUES (?,?,?,?,?,?)",
                  (username.strip(), hash_password(password), (full_name or "").strip(), (email or "").strip(), department_id, now_str()))
        conn.commit(); conn.close()
        return True, f"New HOD '{username}' added successfully."
    except Exception as e:
        return False, f"Database error: {e}"

def list_hods():
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT h.hod_id AS ID, h.username AS Username, h.full_name AS Name, h.email AS Email,
               d.name AS Department, h.created_datetime AS Registered
        FROM hod h LEFT JOIN department d ON d.department_id = h.department_id
        ORDER BY d.name, h.username
    """, conn)
    conn.close()
    return df

def delete_hod(hod_id):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT username FROM hod WHERE hod_id=?", (int(hod_id),))
    row = c.fetchone()
    if not row:
        conn.close()
        return "HOD not found."
    c.execute("DELETE FROM hod WHERE hod_id=?", (int(hod_id),))
    conn.commit(); conn.close()
    return f"HOD '{row[0]}' removed."

# ==================== SOCIETY LOGIC ====================

def add_society(name, department_id):
    if not name or not name.strip():
        return False, "Please enter a society name."
    if not department_id:
        return False, "Please select a department for this society."
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("SELECT society_id FROM society WHERE name=?", (name.strip(),))
        if c.fetchone():
            conn.close()
            return False, "This society already exists."
        c.execute("INSERT INTO society (name, department_id, created_datetime) VALUES (?,?,?)",
                  (name.strip(), department_id, now_str()))
        conn.commit(); conn.close()
        return True, f"Society '{name}' added successfully."
    except Exception as e:
        return False, f"Database error: {e}"

def list_societies(department_id=None):
    conn = get_conn()
    if department_id:
        df = pd.read_sql_query("""
            SELECT s.society_id AS ID, s.name AS Name, d.name AS Department, s.created_datetime AS Created
            FROM society s LEFT JOIN department d ON d.department_id = s.department_id
            WHERE s.department_id = ? ORDER BY s.name
        """, conn, params=(department_id,))
    else:
        df = pd.read_sql_query("""
            SELECT s.society_id AS ID, s.name AS Name, d.name AS Department, s.created_datetime AS Created
            FROM society s LEFT JOIN department d ON d.department_id = s.department_id
            ORDER BY d.name, s.name
        """, conn)
    conn.close()
    return df

def get_society_options(department_id=None):
    df = list_societies(department_id)
    return {row.Name: row.ID for row in df.itertuples()}

# ==================== NOTICE BOARD LOGIC ====================

def add_notice(title, body, posted_by, department_id=None):
    if not title or not title.strip():
        return False, "Please enter a notice title."
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("INSERT INTO notice (title, body, posted_by, department_id, created_datetime) VALUES (?,?,?,?,?)",
                  (title.strip(), (body or "").strip(), posted_by, department_id, now_str()))
        conn.commit(); conn.close()
        return True, "Notice posted to the board."
    except Exception as e:
        return False, f"Database error: {e}"

def list_notices(limit=20):
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT n.notice_id, n.title, n.body, n.posted_by, n.created_datetime, d.name AS department_name
        FROM notice n LEFT JOIN department d ON d.department_id = n.department_id
        ORDER BY n.notice_id DESC LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df

def list_department_notices(department_id, limit=50):
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT n.notice_id, n.title, n.body, n.posted_by, n.created_datetime, d.name AS department_name
        FROM notice n LEFT JOIN department d ON d.department_id = n.department_id
        WHERE n.department_id = ?
        ORDER BY n.notice_id DESC LIMIT ?
    """, conn, params=(department_id, limit))
    conn.close()
    return df

def delete_notice(notice_id):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT title FROM notice WHERE notice_id=?", (int(notice_id),))
    row = c.fetchone()
    if not row:
        conn.close()
        return "Notice not found."
    c.execute("DELETE FROM notice WHERE notice_id=?", (int(notice_id),))
    conn.commit(); conn.close()
    return f"Notice '{row[0]}' deleted."

def is_notice_new(created_datetime_str, days=3):
    try:
        created = datetime.strptime(created_datetime_str, "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - created).days < days
    except Exception:
        return False

def style_status_df(df):
    """Colour the Status column: pending=orange, approved=green, rejected=red."""
    if df.empty or "Status" not in df.columns:
        return df

    def _color(val):
        if val == "approved":
            return "background-color:#16A34A;color:white;font-weight:600;"
        elif val == "rejected":
            return "background-color:#DC2626;color:white;font-weight:600;"
        elif val == "pending":
            return "background-color:#F59E0B;color:white;font-weight:600;"
        return ""

    styler = df.style
    try:
        return styler.map(_color, subset=["Status"])
    except AttributeError:
        return styler.applymap(_color, subset=["Status"])

# ==================== APPLICATION LOGIC ====================

def load_pending_candidates(department_id=None):
    conn = get_conn()
    if department_id:
        df = pd.read_sql_query("""
            SELECT c.candidate_id AS ID, c.name AS Name, c.email AS Email, c.cnic AS CNIC,
                   c.roll_number AS RollNumber, c.party_name AS Society, d.name AS Department, c.status AS Status
            FROM candidate c
            LEFT JOIN society s ON s.society_id = c.society_id
            LEFT JOIN department d ON d.department_id = s.department_id
            WHERE c.status='pending' AND s.department_id = ?
        """, conn, params=(department_id,))
    else:
        df = pd.read_sql_query("""
            SELECT c.candidate_id AS ID, c.name AS Name, c.email AS Email, c.cnic AS CNIC,
                   c.roll_number AS RollNumber, c.party_name AS Society, d.name AS Department, c.status AS Status
            FROM candidate c
            LEFT JOIN society s ON s.society_id = c.society_id
            LEFT JOIN department d ON d.department_id = s.department_id
            WHERE c.status='pending'
        """, conn)
    conn.close()
    return df

def load_all_candidates(department_id=None):
    conn = get_conn()
    if department_id:
        df = pd.read_sql_query("""
            SELECT c.candidate_id AS ID, c.name AS Name, c.cnic AS CNIC, c.roll_number AS RollNumber,
                   c.party_name AS Society, d.name AS Department, c.status AS Status
            FROM candidate c
            LEFT JOIN society s ON s.society_id = c.society_id
            LEFT JOIN department d ON d.department_id = s.department_id
            WHERE s.department_id = ?
        """, conn, params=(department_id,))
    else:
        df = pd.read_sql_query("""
            SELECT c.candidate_id AS ID, c.name AS Name, c.cnic AS CNIC, c.roll_number AS RollNumber,
                   c.party_name AS Society, d.name AS Department, c.status AS Status
            FROM candidate c
            LEFT JOIN society s ON s.society_id = c.society_id
            LEFT JOIN department d ON d.department_id = s.department_id
        """, conn)
    conn.close()
    return df

def approve_candidate(candidate_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT name, email FROM candidate WHERE candidate_id=? AND status='pending'", (int(candidate_id),))
    row = c.fetchone()
    if not row:
        conn.close()
        return "No pending candidate found with that ID."
    name, email = row
    c.execute("UPDATE candidate SET status='approved' WHERE candidate_id=?", (int(candidate_id),))
    conn.commit()
    conn.close()
    sent, email_msg = send_email(
        email, "Candidacy Approved - Election Committee",
        f"Dear {name},\n\nCongratulations! Your candidacy has been approved.\n\nRegards,\nElection Committee")
    # NEW: tell the HOD of this candidate's department that the student was selected
    hod_msg = notify_hod_candidate_approved(candidate_id)
    return f"Candidate '{name}' approved. {email_msg} {hod_msg}"

def reject_candidate(candidate_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT name, email FROM candidate WHERE candidate_id=? AND status='pending'", (int(candidate_id),))
    row = c.fetchone()
    if not row:
        conn.close()
        return "No pending candidate found with that ID."
    name, email = row
    c.execute("UPDATE candidate SET status='rejected' WHERE candidate_id=?", (int(candidate_id),))
    conn.commit()
    conn.close()
    sent, email_msg = send_email(
        email, "Candidacy Update - Election Committee",
        f"Dear {name},\n\nWe regret to inform you that your candidacy application was not approved.\n\nRegards,\nElection Committee")
    return f"Candidate '{name}' rejected. {email_msg}"

def delete_candidate(candidate_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT name FROM candidate WHERE candidate_id=?", (int(candidate_id),))
    row = c.fetchone()
    if not row:
        conn.close()
        return "Candidate not found."
    name = row[0]
    c.execute("DELETE FROM vote WHERE candidate_id=?", (int(candidate_id),))
    c.execute("DELETE FROM candidate WHERE candidate_id=?", (int(candidate_id),))
    conn.commit()
    conn.close()
    return f"Candidate '{name}' permanently deleted."

def register_candidate(name, email, roll_number, cnic, society_id, image):
    if not name or not email or not roll_number or not cnic:
        return False, "Please fill Name, Email, CNIC, and Roll Number."
    valid, digits = is_valid_cnic(cnic)
    if not valid:
        return False, "CNIC must contain 13 digits (e.g. 3630112345678)."
    if not society_id:
        return False, "Please select your society first."
    if not is_email_verified(email):
        return False, "Please verify your email address with the OTP code before registering."
    encoding, face, msg = extract_face_encoding(image)
    if encoding is None:
        return False, msg
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT candidate_id FROM candidate WHERE email=?", (email.strip(),))
        if c.fetchone():
            conn.close()
            return False, "A candidate with this email is already registered."
        cnic_hash = hash_cnic(digits)
        c.execute("SELECT candidate_id FROM candidate WHERE cnic_hash=?", (cnic_hash,))
        if c.fetchone():
            conn.close()
            return False, "A candidate with this CNIC is already registered."
        c.execute("SELECT name FROM society WHERE society_id=?", (society_id,))
        soc_row = c.fetchone()
        society_name = soc_row[0] if soc_row else ""
        c.execute("""INSERT INTO candidate (name, email, roll_number, cnic, cnic_hash, party_name, society_id, status, email_verified, registered_datetime)
                     VALUES (?,?,?,?,?,?,?,?,?,?)""",
                  (name.strip(), email.strip(), roll_number.strip(), mask_cnic(digits), cnic_hash, society_name,
                   society_id, "pending", 1, now_str()))
        new_candidate_id = c.lastrowid
        conn.commit()
        conn.close()
    except Exception as e:
        return False, f"Database error: {e}"
    # NEW: email Super Admin(s) + Dean(s) asking them to approve this candidate
    notify_msg = notify_admins_new_candidate(new_candidate_id)
    return True, f"Candidate '{name}' registered in society '{society_name}'. Awaiting admin approval. {notify_msg}"

def register_voter(cnic, name, roll_number, society_id, image):
    if not cnic or not name or not roll_number:
        return False, "Please fill CNIC, Name, and Roll Number."
    if not society_id:
        return False, "Please select the voter's society first."
    valid, digits = is_valid_cnic(cnic)
    if not valid:
        return False, "CNIC must contain 13 digits (e.g. 3630112345678)."
    encoding, face, msg = extract_face_encoding(image)
    if encoding is None:
        return False, msg
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT voter_id FROM voter WHERE cnic_hash=?", (hash_cnic(digits),))
        if c.fetchone():
            conn.close()
            return False, "A voter with this CNIC is already registered."
        c.execute("""INSERT INTO voter (cnic, cnic_hash, name, roll_number, voting_status, face_image, face_encoding, society_id, registered_datetime)
                     VALUES (?,?,?,?,?,?,?,?,?)""",
                  (mask_cnic(digits), hash_cnic(digits), name.strip(), roll_number.strip(), 0, face_to_blob(face),
                   encoding_to_blob(encoding), society_id, now_str()))
        conn.commit()
        conn.close()
    except Exception as e:
        return False, f"Database error: {e}"
    return True, f"Voter '{name}' registered successfully (CNN facial template stored). Go to the 'Vote' tab to continue."

def get_approved_candidates_df(society_id=None):
    conn = get_conn()
    if society_id:
        df = pd.read_sql_query(
            "SELECT candidate_id AS ID, name AS Candidate, roll_number AS RollNumber, party_name AS Society "
            "FROM candidate WHERE status='approved' AND society_id=?", conn, params=(society_id,))
    else:
        df = pd.read_sql_query(
            "SELECT candidate_id AS ID, name AS Candidate, roll_number AS RollNumber, party_name AS Society "
            "FROM candidate WHERE status='approved'", conn)
    conn.close()
    return df

def verify_voter_face(cnic, frame_open, frame_blink):
    if not cnic:
        return False, "Enter your CNIC first.", None
    valid, digits = is_valid_cnic(cnic)
    if not valid:
        return False, "CNIC must contain 13 digits.", None

    probe_encoding, face_check, qmsg = extract_face_encoding(frame_open)
    if probe_encoding is None:
        return False, qmsg, None

    live_ok, live_msg = liveness_blink_check(frame_open, frame_blink)

    conn = get_conn()
    c = conn.cursor()
    c.execute("""SELECT v.voter_id, v.name, v.voting_status, v.society_id, s.name, v.face_encoding
                 FROM voter v LEFT JOIN society s ON s.society_id = v.society_id
                 WHERE v.cnic_hash=?""", (hash_cnic(digits),))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "CNIC not found. Please register first in the 'Polling Officer' tab.", None

    voter_id, name, voting_status, society_id, society_name, stored_encoding_blob = row
    if voting_status == 1:
        conn.close()
        return False, f"{name}, you have already cast your vote. One Person - One Vote.", None

    if not live_ok:
        c.execute("""INSERT INTO voter_authentication_log
                     (voter_id, datetime, auth_method, auth_status, face_match_confidence, liveness_status)
                     VALUES (?,?,?,?,?,?)""",
                  (voter_id, now_str(), "Face Recognition (CNN)", "Fail", 0.0, "Fail"))
        conn.commit()
        conn.close()
        return False, live_msg, None

    if not stored_encoding_blob:
        conn.close()
        return False, "No biometric template found for this voter. Please contact the Polling Officer.", None

    stored_encoding = blob_to_encoding(stored_encoding_blob)
    # Both embeddings are L2-normalized, so cosine similarity is just the dot product.
    similarity = float(np.dot(stored_encoding, probe_encoding))
    match = similarity >= FACE_MATCH_MIN_SIMILARITY
    auth_status = "Success" if match else "Fail"

    c.execute("""INSERT INTO voter_authentication_log
                 (voter_id, datetime, auth_method, auth_status, face_match_confidence, liveness_status)
                 VALUES (?,?,?,?,?,?)""",
              (voter_id, now_str(), "Face Recognition (CNN - YOLOv8-face + ArcFace)", auth_status, similarity, "Pass"))
    auth_log_id = c.lastrowid
    conn.commit()
    conn.close()

    if not match:
        return False, f"Face does not match our records for CNIC {digits}. Access denied.", None
    return True, (f"{live_msg} Face verified for {name} "
                   f"(CNN embedding similarity {similarity:.3f}, higher = better; threshold {FACE_MATCH_MIN_SIMILARITY})."), \
           {"voter_id": voter_id, "name": name, "auth_log_id": auth_log_id,
            "society_id": society_id, "society_name": society_name}

def verify_location(voter_state, lat, lon):
    within, dist = gps_check(lat, lon)
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO gps_radius_check_log
                 (auth_log_id, captured_lat, captured_lon, distance_to_boundary, within_radius)
                 VALUES (?,?,?,?,?)""",
              (voter_state["auth_log_id"], lat, lon, dist, 1 if within else 0))
    conn.commit()
    conn.close()
    if not within:
        return False, (f"You are {dist:.0f}m from the voting boundary "
                        f"(allowed radius {ALLOWED_RADIUS_METERS}m). Access denied - outside authorized voting area.")
    return True, (f"Location verified ({dist:.0f}m from center, within {ALLOWED_RADIUS_METERS}m radius). "
                  f"You may now cast your vote.")

def cast_vote(voter_state, candidate_id):
    voter_id = voter_state["voter_id"]
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT voting_status FROM voter WHERE voter_id=?", (voter_id,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return False, "Voter record not found.", None
    if row[0] == 1:
        conn.close()
        return False, "Vote rejected - this voter has already voted (One Person, One Vote).", None

    # Tamper-evident hash chain: each vote's hash includes the previous vote's
    # hash, so any later edit/delete/reorder of a past row breaks the chain.
    c.execute("SELECT record_hash FROM vote ORDER BY vote_id DESC LIMIT 1")
    prev_row = c.fetchone()
    prev_hash = prev_row[0] if prev_row and prev_row[0] else GENESIS_HASH
    timestamp = now_str()
    record_hash = _compute_vote_hash(voter_id, candidate_id, timestamp, prev_hash)
    receipt_code = generate_receipt_code()

    c.execute("""INSERT INTO vote (voter_id, candidate_id, vote_timestamp, prev_hash, record_hash, receipt_code)
                 VALUES (?,?,?,?,?,?)""",
              (voter_id, candidate_id, timestamp, prev_hash, record_hash, receipt_code))
    c.execute("UPDATE voter SET voting_status=1 WHERE voter_id=?", (voter_id,))
    conn.commit()
    conn.close()
    return True, f"Vote recorded successfully for {voter_state['name']}. Thank you for voting!", receipt_code

def verify_vote_chain():
    """
    Re-walks every vote in insertion order and recomputes each record_hash
    from (voter_id, candidate_id, timestamp, prev_hash), comparing it to what
    is stored. Returns (is_valid, total_checked, first_broken_vote_id_or_None).
    Detects any tampering (edited/deleted/reordered vote rows) - proof of
    NFR-7 Data Integrity for the SRS/FYP defense.
    """
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT vote_id, voter_id, candidate_id, vote_timestamp, prev_hash, record_hash "
        "FROM vote ORDER BY vote_id ASC", conn)
    conn.close()
    if df.empty:
        return True, 0, None
    expected_prev = GENESIS_HASH
    for row in df.itertuples():
        if row.prev_hash != expected_prev:
            return False, len(df), row.vote_id
        recomputed = _compute_vote_hash(row.voter_id, row.candidate_id, row.vote_timestamp, row.prev_hash)
        if recomputed != row.record_hash:
            return False, len(df), row.vote_id
        expected_prev = row.record_hash
    return True, len(df), None

def verify_receipt_code(receipt_code):
    """Voter-verifiable receipt lookup: confirms a vote with this code exists
    and when it was cast, WITHOUT revealing which candidate was chosen
    (preserves ballot secrecy)."""
    if not receipt_code:
        return False, "Enter your receipt code."
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT vote_timestamp FROM vote WHERE receipt_code=?", (receipt_code.strip().upper(),))
    row = c.fetchone()
    conn.close()
    if not row:
        return False, "No vote found for this receipt code."
    return True, f"Verified: a vote was recorded at {row[0]}. Your choice remains confidential."

def get_results_df():
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT c.name AS Candidate, dpt.name AS Department, c.party_name AS Society, COUNT(v.vote_id) AS Votes
        FROM candidate c
        LEFT JOIN vote v ON v.candidate_id = c.candidate_id
        LEFT JOIN society s ON s.society_id = c.society_id
        LEFT JOIN department dpt ON dpt.department_id = s.department_id
        WHERE c.status='approved'
        GROUP BY c.candidate_id
        ORDER BY Department, Society, Votes DESC
    """, conn)
    conn.close()
    return df

def get_society_winners_df():
    df = get_results_df()
    if df.empty:
        return df
    idx = df.groupby("Society")["Votes"].idxmax()
    winners = df.loc[idx].sort_values("Society").reset_index(drop=True)
    return winners

def get_turnout_stats():
    conn = get_conn()
    total_voters = pd.read_sql_query("SELECT COUNT(*) AS n FROM voter", conn).iloc[0]["n"]
    voted = pd.read_sql_query("SELECT COUNT(*) AS n FROM voter WHERE voting_status=1", conn).iloc[0]["n"]
    conn.close()
    turnout = f"{voted}/{total_voters} registered voters have voted"
    if total_voters:
        turnout += f" ({100*voted/total_voters:.1f}%)"
    return turnout

def get_auth_logs():
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT a.auth_log_id AS LogID, v.name AS Voter, a.datetime AS Time,
               a.auth_status AS FaceStatus, a.liveness_status AS Liveness,
               ROUND(a.face_match_confidence, 3) AS CNN_Similarity
        FROM voter_authentication_log a
        LEFT JOIN voter v ON v.voter_id = a.voter_id
        ORDER BY a.auth_log_id DESC
    """, conn)
    conn.close()
    return df

# ==================== STREAMLIT UI ====================

st.set_page_config(page_title=f"{ORG_NAME} - E-Voting Portal", layout="wide")

def is_lan_request():
    try:
        ip = st.context.ip_address
    except Exception:
        ip = None
    if ip is None:
        return True
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private or addr.is_loopback
    except ValueError:
        return False

if not is_lan_request():
    st.error("Access denied. This system is restricted to the intranet (local network) only.")
    st.stop()

st.markdown("""
<style>
    .stApp { background-color: #F4F5F9 !important; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background-color: #F4F5F9 !important; }
    html, body, [class*="css"] { color: #1A1A2E !important; }

    #header-banner {
        background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 100%);
        padding: 28px 28px; border-radius: 14px; margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.25); text-align:center;
    }
    #header-banner h1, #header-banner p { color: #FFFFFF !important; margin: 0; }
    #header-banner h1 { font-size: 2rem; font-weight: 700; }
    #header-banner p { font-size: 0.95rem; opacity: 0.9; margin-top: 6px; letter-spacing: 1px; }

    [data-testid="stVerticalBlockBorderWrapper"] { background-color: #FFFFFF; border-radius: 10px; }

    h1, h2, h3, h4, h5, h6, label, p, span, .stMarkdown { color: #1A1A2E !important; }
    [data-testid="stCaptionContainer"] { color: #5A6072 !important; }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important; color: #1A1A2E !important;
        border: 1px solid #CBD2E1 !important; border-radius: 8px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #2563EB !important; box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }

    .stButton>button {
        background-color: #2563EB; color: #FFFFFF !important; border: none;
        border-radius: 8px; font-weight: 600; padding: 0.5rem 1.2rem;
        transition: background-color 0.15s ease-in-out;
    }
    .stButton>button:hover { background-color: #1E3A8A; color: #FFFFFF !important; }
    .stButton>button p { color: #FFFFFF !important; }

    [data-testid="stDataFrame"] { border: 1px solid #E2E5EE; border-radius: 8px; overflow: hidden; }

    [data-testid="stAlertContentSuccess"] p, [data-testid="stAlertContentWarning"] p,
    [data-testid="stAlertContentError"] p, [data-testid="stAlertContentInfo"] p { color: #1A1A2E !important; }

    [data-testid="stCameraInput"] {
        border: 1px solid #E2E5EE; border-radius: 10px; padding: 8px; background-color: #FFFFFF;
    }

    /* ---- Portal cards on the Home page: title/subtitle + Open button now
       live INSIDE one bordered st.container, so the border wraps both. ---- */
    .portal-title { font-size:1.1rem; font-weight:700; color:#1A1A2E; margin-bottom:4px; }
    .portal-subtitle { font-size:0.88rem; color:#5A6072; }

    .role-pill {
        display:inline-block; padding:3px 12px; border-radius:12px; font-size:0.72rem;
        font-weight:700; letter-spacing:0.5px; margin-left:8px; vertical-align:middle;
    }
    .role-pill-superadmin { background:#7C3AED; color:#FFF; }
    .role-pill-dean { background:#2563EB; color:#FFF; }
    .role-pill-hod { background:#0EA5E9; color:#FFF; }
</style>
<div id="header-banner">
    <h1>""" + ORG_NAME + """</h1>
    <p>E-VOTING PORTAL - CNN FACE RECOGNITION - BLINK LIVENESS - GPS VERIFIED</p>
</div>
""", unsafe_allow_html=True)

if not CASCADES_OK:
    st.error(
        "Face detection cascade files could not be loaded. The app tried to download them "
        "automatically but that failed - please check your internet connection and restart the app."
    )
if not YOLO_OK:
    st.warning(
        "YOLOv8-face weights could not be downloaded (needs internet on first run). "
        "Falling back to the Haar cascade for face detection until this is resolved."
    )
if not INSIGHTFACE_OK:
    st.error(
        "InsightFace ArcFace model could not be loaded (needs internet on first run to download "
        "its model pack to ~/.insightface/models/). Face registration and voting will not work "
        "until this is resolved - please restart the app with an internet connection."
    )

for key, default in [
    ("superadmin_authenticated", False), ("superadmin_name", ""),
    ("dean_authenticated", False), ("dean_name", ""),
    ("hod_authenticated", False), ("hod_name", ""), ("hod_department_id", None), ("hod_department_name", ""),
    ("voter_state", None), ("face_ok", False), ("gps_ok", False), ("page", "Home"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------- PAGE: HOME (rows menu) --------------------

def render_notice_board():
    st.markdown("""
    <style>
    .notice-card {
        background:#FFFFFF; border:1px solid #E2E5EE; border-radius:10px;
        padding:14px 18px; margin-bottom:10px; box-shadow:0 2px 6px rgba(0,0,0,0.04);
    }
    .notice-title { font-weight:700; font-size:1rem; color:#1A1A2E; }
    .notice-new-badge {
        background:#DC2626; color:white; font-size:0.68rem; font-weight:700;
        padding:2px 8px; border-radius:10px; margin-left:8px; letter-spacing:0.5px;
        vertical-align:middle;
    }
    .notice-meta { color:#8A90A2; font-size:0.78rem; margin-top:2px; }
    .notice-body { color:#3A3F52; font-size:0.9rem; margin-top:8px; white-space:pre-wrap; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### Notice Board")
    notices_df = list_notices()
    if notices_df.empty:
        st.info("No notices posted yet.")
        return

    for row in notices_df.itertuples():
        dept_tag = f" ({row.department_name})" if getattr(row, "department_name", None) else " (Institute-wide)"
        expander_label = row.title + dept_tag + ("  [NEW]" if is_notice_new(row.created_datetime) else "")
        with st.expander(expander_label, expanded=False):
            # FIX: html.escape() so an admin can't inject HTML/scripts into a notice
            st.markdown(
                f"<div class='notice-meta'>Posted by {html.escape(str(row.posted_by or 'Election Committee'))} - {row.created_datetime}</div>"
                f"<div class='notice-body'>{html.escape(str(row.body or ''))}</div>",
                unsafe_allow_html=True)


def render_home():
    render_notice_board()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Portals")
    st.markdown("<p style='color:#5A6072;'>Select a portal below</p>", unsafe_allow_html=True)

    cards = [
        ("Admin", "Admin Panel", "Super Admin, Dean & HOD login - manage departments, societies, candidates and notices"),
        ("Candidate Registration", "Candidate Registration", "Register as a candidate on behalf of your society"),
        ("Polling Officer", "Polling Officer", "Register voters, cast votes, and monitor turnout/logs (Polling Officer duty)"),
        ("Results", "Results", "View live election results and society-wise winners"),
    ]

    button_colors = ["#7C3AED,#9333EA", "#16A34A,#22C55E", "#0EA5E9,#38BDF8", "#1E3A8A,#2563EB"]

    # NOTE: title/subtitle AND the "Open" button are now rendered inside the
    # SAME st.container(border=True), so the visible white border wraps
    # around both the text and the button (instead of the button sitting
    # outside the card, in its own column with no border).
    for i, (key, title, subtitle) in enumerate(cards):
        with st.container(border=True):
            row_l, row_r = st.columns([4, 1], vertical_alignment="center")
            with row_l:
                st.markdown(
                    f"<div class='portal-title'>{title}</div>"
                    f"<div class='portal-subtitle'>{subtitle}</div>",
                    unsafe_allow_html=True)
            with row_r:
                st.markdown(
                    f"<style>div[data-testid='stButton']:has(button[kind]) button#nav_btn_{i} "
                    f"{{background:linear-gradient(90deg,{button_colors[i % len(button_colors)]}) !important;}}</style>",
                    unsafe_allow_html=True)
                if st.button("Open", key=f"nav_{key}", use_container_width=True):
                    st.session_state["page"] = key
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

# -------------------- ADMIN HUB: role selector + 3 dashboards --------------------

def render_admin_hub_page():
    if st.session_state["superadmin_authenticated"]:
        render_superadmin_dashboard()
        return
    if st.session_state["dean_authenticated"]:
        render_dean_dashboard()
        return
    if st.session_state["hod_authenticated"]:
        render_hod_dashboard()
        return

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.container(border=True):
            st.markdown("### Admin Panel Login")
            st.caption("Select your role and sign in")
            role = st.radio("Role", ["Super Admin", "Dean", "HOD"], horizontal=True, key="admin_role_select")
            uname = st.text_input("Username", key="hub_uname")
            pw = st.text_input("Password", type="password", key="hub_pw")
            if st.button("Sign In", key="hub_signin_btn", use_container_width=True):
                role_key = {"Super Admin": "superadmin", "Dean": "dean", "HOD": "hod"}[role]
                is_locked, remaining = check_lockout(role_key, uname)
                if is_locked:
                    st.error(f"Too many failed attempts. Locked for {remaining} more second(s) - please wait.")
                elif role == "Super Admin":
                    row = authenticate_superadmin(uname, pw)
                    record_login_result(role_key, uname, bool(row))
                    if row:
                        st.session_state["superadmin_authenticated"] = True
                        st.session_state["superadmin_name"] = row[1] or uname
                        st.rerun()
                    else:
                        st.error("Incorrect Super Admin username or password.")
                elif role == "Dean":
                    row = authenticate_dean(uname, pw)
                    record_login_result(role_key, uname, bool(row))
                    if row:
                        st.session_state["dean_authenticated"] = True
                        st.session_state["dean_name"] = row[1] or uname
                        st.rerun()
                    else:
                        st.error("Incorrect Dean username or password.")
                else:
                    row = authenticate_hod(uname, pw)
                    record_login_result(role_key, uname, bool(row))
                    if row:
                        st.session_state["hod_authenticated"] = True
                        st.session_state["hod_name"] = row[1] or uname
                        st.session_state["hod_department_id"] = row[2]
                        st.session_state["hod_department_name"] = row[3]
                        st.rerun()
                    else:
                        st.error("Incorrect HOD username or password.")
            st.markdown("<hr>", unsafe_allow_html=True)
            st.caption("Dean and HOD accounts are created by the Super Admin from the Super Admin dashboard. "
                       "Default Super Admin login: **superadmin / super123** (please change this after first login).")


def _render_notice_management(notices_df, can_delete_prefix):
    if notices_df.empty:
        st.info("No notices posted yet.")
        return
    with st.container(border=True):
        header_cols = st.columns([0.6, 3, 2, 2, 1])
        for col, h in zip(header_cols, ["ID", "Title", "Scope", "Posted", ""]):
            col.markdown(
                f"<p style='font-weight:700;color:#5A6072;font-size:0.8rem;"
                f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0;'>{h}</p>",
                unsafe_allow_html=True)
        st.markdown("<hr style='margin:6px 0 10px 0;border-color:#E2E5EE;'>", unsafe_allow_html=True)
        for i, row in enumerate(notices_df.itertuples()):
            cols = st.columns([0.6, 3, 2, 2, 1])
            cols[0].markdown(
                f"<span style='background:#EEF2FF;color:#1E3A8A;padding:3px 10px;"
                f"border-radius:6px;font-weight:600;font-size:0.85rem;'>{row.notice_id}</span>",
                unsafe_allow_html=True)
            cols[1].markdown(f"<b>{html.escape(str(row.title))}</b>", unsafe_allow_html=True)
            cols[2].write(row.department_name or "Institute-wide")
            cols[3].write(row.created_datetime)
            with cols[4]:
                if st.button("Delete", key=f"{can_delete_prefix}_del_notice_{row.notice_id}", use_container_width=True):
                    st.info(delete_notice(row.notice_id))
                    st.rerun()
            if i < len(notices_df) - 1:
                st.markdown("<hr style='margin:10px 0;border-color:#F0F1F5;'>", unsafe_allow_html=True)


def _render_society_management(department_id, key_prefix, fixed_department=False):
    st.markdown("#### Societies")
    dept_options = get_department_options()
    soc_col1, soc_col2, soc_col3 = st.columns([2, 2, 1])
    with soc_col1:
        new_soc = st.text_input("New society name", key=f"{key_prefix}_new_society_input")
    if fixed_department:
        target_dept_id = department_id
        with soc_col2:
            st.text_input("Department", value=st.session_state.get("hod_department_name", ""), disabled=True,
                          key=f"{key_prefix}_soc_dept_display")
    else:
        with soc_col2:
            if not dept_options:
                st.warning("Add a department first.")
                target_dept_id = None
            else:
                soc_dept_name = st.selectbox("Department", list(dept_options.keys()), key=f"{key_prefix}_soc_dept_select")
                target_dept_id = dept_options[soc_dept_name]
    with soc_col3:
        st.write("")
        if st.button("Add Society", key=f"{key_prefix}_add_society_btn"):
            ok, msg = add_society(new_soc, target_dept_id)
            (st.success if ok else st.warning)(msg)
    st.dataframe(list_societies(department_id), width="stretch")


def _render_candidate_management(department_id, key_prefix):
    if st.button("Refresh List", key=f"{key_prefix}_refresh_cands"):
        st.rerun()

    st.markdown("#### Pending Candidate Approvals")
    pending_df = load_pending_candidates(department_id)
    if pending_df.empty:
        st.info("No pending candidates.")
    else:
        with st.container(border=True):
            header_cols = st.columns([0.6, 2, 1.6, 1.6, 1.3, 1.3])
            headers = ["ID", "Name", "Roll No", "Society", "", ""]
            for col, h in zip(header_cols, headers):
                col.markdown(
                    f"<p style='font-weight:700;color:#5A6072;font-size:0.8rem;"
                    f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0;'>{h}</p>",
                    unsafe_allow_html=True)
            st.markdown("<hr style='margin:6px 0 10px 0;border-color:#E2E5EE;'>", unsafe_allow_html=True)

            for i, row in enumerate(pending_df.itertuples()):
                cols = st.columns([0.6, 2, 1.6, 1.6, 1.3, 1.3])
                cols[0].markdown(
                    f"<span style='background:#EEF2FF;color:#1E3A8A;padding:3px 10px;"
                    f"border-radius:6px;font-weight:600;font-size:0.85rem;'>{row.ID}</span>",
                    unsafe_allow_html=True)
                cols[1].markdown(f"<b>{html.escape(str(row.Name))}</b>", unsafe_allow_html=True)
                cols[2].write(row.RollNumber)
                cols[3].write(row.Society)
                with cols[4]:
                    if st.button("Approve", key=f"{key_prefix}_approve_{row.ID}", use_container_width=True):
                        st.info(approve_candidate(row.ID))
                        st.rerun()
                with cols[5]:
                    if st.button("Reject", key=f"{key_prefix}_reject_{row.ID}", use_container_width=True):
                        st.info(reject_candidate(row.ID))
                        st.rerun()
                if i < len(pending_df) - 1:
                    st.markdown("<hr style='margin:10px 0;border-color:#F0F1F5;'>", unsafe_allow_html=True)

    st.markdown("#### All Candidates")
    all_df = load_all_candidates(department_id)
    if all_df.empty:
        st.info("No candidates registered.")
    else:
        status_colors = {"approved": "#16A34A", "rejected": "#DC2626", "pending": "#F59E0B"}
        with st.container(border=True):
            header_cols = st.columns([0.6, 2, 1.8, 1.4, 1])
            headers = ["ID", "Name", "Society", "Status", ""]
            for col, h in zip(header_cols, headers):
                col.markdown(
                    f"<p style='font-weight:700;color:#5A6072;font-size:0.8rem;"
                    f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0;'>{h}</p>",
                    unsafe_allow_html=True)
            st.markdown("<hr style='margin:6px 0 10px 0;border-color:#E2E5EE;'>", unsafe_allow_html=True)

            for i, row in enumerate(all_df.itertuples()):
                cols = st.columns([0.6, 2, 1.8, 1.4, 1])
                cols[0].markdown(
                    f"<span style='background:#EEF2FF;color:#1E3A8A;padding:3px 10px;"
                    f"border-radius:6px;font-weight:600;font-size:0.85rem;'>{row.ID}</span>",
                    unsafe_allow_html=True)
                cols[1].markdown(f"<b>{html.escape(str(row.Name))}</b>", unsafe_allow_html=True)
                cols[2].write(row.Society)
                color = status_colors.get(row.Status, "#9CA3AF")
                cols[3].markdown(
                    f"<span style='background-color:{color};color:white;padding:3px 12px;"
                    f"border-radius:6px;font-weight:600;font-size:0.8rem;'>{row.Status}</span>",
                    unsafe_allow_html=True)
                with cols[4]:
                    if st.button("Delete", key=f"{key_prefix}_delete_{row.ID}", use_container_width=True):
                        st.info(delete_candidate(row.ID))
                        st.rerun()
                if i < len(all_df) - 1:
                    st.markdown("<hr style='margin:10px 0;border-color:#F0F1F5;'>", unsafe_allow_html=True)


def render_superadmin_dashboard():
    st.markdown(f"<h4>Logged in as <span class='role-pill role-pill-superadmin'>SUPER ADMIN</span> "
                f"&nbsp; {html.escape(str(st.session_state['superadmin_name']))}</h4>", unsafe_allow_html=True)
    if st.button("Logout", key="superadmin_logout_btn"):
        st.session_state["superadmin_authenticated"] = False
        st.rerun()

    st.markdown("#### Departments")
    d1, d2 = st.columns([3, 1])
    with d1:
        new_dept = st.text_input("New department name", key="sa_new_dept_input")
    with d2:
        st.write("")
        if st.button("Add Department", key="sa_add_dept_btn"):
            ok, msg = add_department(new_dept)
            (st.success if ok else st.warning)(msg)
    st.dataframe(list_departments(), width="stretch")

    st.markdown("#### Deans (Institute-wide Admins)")
    with st.expander("Add a New Dean"):
        de_u = st.text_input("Dean Username", key="sa_new_dean_uname")
        de_n = st.text_input("Dean Full Name", key="sa_new_dean_fullname")
        de_e = st.text_input("Dean Email (approval requests are sent here)", key="sa_new_dean_email")
        de_p = st.text_input("Dean Password", type="password", key="sa_new_dean_pw")
        if st.button("Add Dean", key="sa_add_dean_btn"):
            ok, msg = add_dean(de_u, de_p, de_n, de_e)
            (st.success if ok else st.warning)(msg)
    deans_df = list_deans()
    st.dataframe(deans_df, width="stretch")
    if not deans_df.empty:
        del_col1, del_col2 = st.columns([3, 1])
        with del_col1:
            del_dean_id = st.selectbox("Remove a Dean", deans_df["ID"].tolist(), key="sa_del_dean_select")
        with del_col2:
            st.write("")
            if st.button("Remove Dean", key="sa_del_dean_btn"):
                st.info(delete_dean(del_dean_id))
                st.rerun()

    st.markdown("#### HODs (Department Admins)")
    dept_options = get_department_options()
    with st.expander("Add a New HOD"):
        if not dept_options:
            st.warning("Please add a department first.")
        else:
            ho_u = st.text_input("HOD Username", key="sa_new_hod_uname")
            ho_n = st.text_input("HOD Full Name", key="sa_new_hod_fullname")
            ho_e = st.text_input("HOD Email (selected-candidate notices are sent here)", key="sa_new_hod_email")
            ho_p = st.text_input("HOD Password", type="password", key="sa_new_hod_pw")
            ho_dept_name = st.selectbox("Department", list(dept_options.keys()), key="sa_new_hod_dept")
            if st.button("Add HOD", key="sa_add_hod_btn"):
                ok, msg = add_hod(ho_u, ho_p, ho_n, dept_options[ho_dept_name], ho_e)
                (st.success if ok else st.warning)(msg)
    hods_df = list_hods()
    st.dataframe(hods_df, width="stretch")
    if not hods_df.empty:
        del_col1, del_col2 = st.columns([3, 1])
        with del_col1:
            del_hod_id = st.selectbox("Remove a HOD", hods_df["ID"].tolist(), key="sa_del_hod_select")
        with del_col2:
            st.write("")
            if st.button("Remove HOD", key="sa_del_hod_btn"):
                st.info(delete_hod(del_hod_id))
                st.rerun()

    st.markdown("#### Notification Emails")
    st.caption("Set or change the email of any Super Admin, Dean or HOD. New-candidate approval requests go to "
               "Super Admins and Deans; 'candidate selected' notices go to the HOD of that department.")
    ne_role = st.selectbox("Account type", ["Super Admin", "Dean", "HOD"], key="sa_ne_role")
    ne_df = {"Super Admin": list_superadmins(), "Dean": list_deans(), "HOD": list_hods()}[ne_role]
    if ne_df.empty:
        st.info(f"No {ne_role} accounts yet.")
    else:
        ne_labels = {f"{r.Username} (ID {r.ID}) - {r.Email or 'no email set'}": r.ID for r in ne_df.itertuples()}
        ne_pick = st.selectbox("Account", list(ne_labels.keys()), key="sa_ne_account")
        ne_email = st.text_input("Email address", key="sa_ne_email")
        if st.button("Save Email", key="sa_ne_save_btn"):
            ok, msg = set_account_email(ne_role, ne_labels[ne_pick], ne_email)
            (st.success if ok else st.warning)(msg)
            if ok:
                st.rerun()

    _render_society_management(None, key_prefix="sa")

    st.markdown("#### Notice Board (Institute-wide)")
    with st.expander("Post a New Notice"):
        n_title = st.text_input("Notice Title", key="sa_n_title")
        n_body = st.text_area("Notice Detail (optional)", key="sa_n_body")
        if st.button("Post Notice", key="sa_add_notice_btn"):
            ok, msg = add_notice(n_title, n_body, st.session_state.get("superadmin_name") or "Super Admin",
                                  department_id=None)
            (st.success if ok else st.warning)(msg)
            if ok:
                st.rerun()
    _render_notice_management(list_notices(), can_delete_prefix="sa")

    st.markdown("#### Candidates (All Departments)")
    _render_candidate_management(None, key_prefix="sa")


def render_dean_dashboard():
    st.markdown(f"<h4>Logged in as <span class='role-pill role-pill-dean'>DEAN</span> "
                f"&nbsp; {html.escape(str(st.session_state['dean_name']))}</h4>", unsafe_allow_html=True)
    if st.button("Logout", key="dean_logout_btn"):
        st.session_state["dean_authenticated"] = False
        st.rerun()

    st.caption("Dean access: full admin over the whole institute (all departments and societies).")

    _render_society_management(None, key_prefix="dean")

    st.markdown("#### Notice Board (Institute-wide)")
    with st.expander("Post a New Notice"):
        n_title = st.text_input("Notice Title", key="dean_n_title")
        n_body = st.text_area("Notice Detail (optional)", key="dean_n_body")
        if st.button("Post Notice", key="dean_add_notice_btn"):
            ok, msg = add_notice(n_title, n_body, st.session_state.get("dean_name") or "Dean",
                                  department_id=None)
            (st.success if ok else st.warning)(msg)
            if ok:
                st.rerun()
    _render_notice_management(list_notices(), can_delete_prefix="dean")

    st.markdown("#### Candidates (All Departments)")
    _render_candidate_management(None, key_prefix="dean")


def render_hod_dashboard():
    dept_id = st.session_state["hod_department_id"]
    dept_name = st.session_state["hod_department_name"] or "N/A"
    st.markdown(f"<h4>Logged in as <span class='role-pill role-pill-hod'>HOD</span> "
                f"&nbsp; {html.escape(str(st.session_state['hod_name']))} &nbsp;-&nbsp; Department: <b>{html.escape(str(dept_name))}</b></h4>",
                unsafe_allow_html=True)
    if st.button("Logout", key="hod_logout_btn"):
        st.session_state["hod_authenticated"] = False
        st.rerun()

    st.caption(f"HOD access: full admin scoped to the '{dept_name}' department only.")

    _render_society_management(dept_id, key_prefix="hod", fixed_department=True)

    st.markdown(f"#### Notice Board ({dept_name})")
    with st.expander("Post a New Notice"):
        n_title = st.text_input("Notice Title", key="hod_n_title")
        n_body = st.text_area("Notice Detail (optional)", key="hod_n_body")
        if st.button("Post Notice", key="hod_add_notice_btn"):
            ok, msg = add_notice(n_title, n_body, st.session_state.get("hod_name") or "HOD",
                                  department_id=dept_id)
            (st.success if ok else st.warning)(msg)
            if ok:
                st.rerun()
    _render_notice_management(list_department_notices(dept_id), can_delete_prefix="hod")

    st.markdown(f"#### Candidates ({dept_name})")
    _render_candidate_management(dept_id, key_prefix="hod")

# -------------------- PAGE: CANDIDATE REGISTRATION --------------------

def render_candidate_page():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.container(border=True):
            st.markdown("### Candidate Registration")
            st.caption("Register as a candidate on behalf of your society")

            dept_options = get_department_options()
            if not dept_options:
                st.warning("No departments have been created yet. Please ask a Super Admin to add one first.")
                return
            c_dept_name = st.selectbox("Select Your Department", list(dept_options.keys()), key="c_dept")
            society_options = get_society_options(dept_options[c_dept_name])
            if not society_options:
                st.warning("No societies exist yet in this department. Please ask an admin to add one first.")
                return

            c_name = st.text_input("Full Name", key="c_name")
            c_email = st.text_input("Email", key="c_email")

            st.markdown("**Email Verification**")
            otp_col1, otp_col2, otp_col3 = st.columns([1.3, 1, 1])
            with otp_col1:
                c_otp = st.text_input("6-digit code sent to your email", key="c_otp_input", max_chars=6)
            with otp_col2:
                st.write("")
                if st.button("Send Code", key="c_send_otp_btn", use_container_width=True):
                    ok, msg = send_candidate_otp(c_email)
                    (st.success if ok else st.warning)(msg)
            with otp_col3:
                st.write("")
                if st.button("Verify Code", key="c_verify_otp_btn", use_container_width=True):
                    ok, msg = verify_candidate_otp(c_email, c_otp)
                    (st.success if ok else st.warning)(msg)
            if is_email_verified(c_email):
                st.success(f"Email verified: {c_email}")
            else:
                st.caption("Enter your email above, send yourself a code, then verify it before submitting.")

            c_cnic = st.text_input("CNIC (13 digits)", key="c_cnic")
            c_roll = st.text_input("University Roll Number", key="c_roll")
            c_society_name = st.selectbox("Select Your Society", list(society_options.keys()), key="c_society")

            c_image = render_auto_face_capture("cand")

            if st.button("Register Candidate", use_container_width=True):
                if c_image is None:
                    st.warning("Please position your face in front of the camera so the photo can be captured.")
                else:
                    ok, msg = register_candidate(c_name, c_email, c_roll, c_cnic, society_options[c_society_name], c_image)
                    (st.success if ok else st.warning)(msg)

# -------------------- PAGE: POLLING OFFICER (hub: voter registration + cast vote + election officer) --------------------

def _render_voter_registration_section():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.container(border=True):
            st.caption("Voter Registration")

            dept_options = get_department_options()
            if not dept_options:
                st.warning("No departments have been created yet. Please ask a Super Admin to add one first.")
                return
            v_dept_name = st.selectbox("Select Voter's Department", list(dept_options.keys()), key="v_dept")
            society_options = get_society_options(dept_options[v_dept_name])
            if not society_options:
                st.warning("No societies exist yet in this department. Please ask an admin to add one first.")
                return

            v_cnic = st.text_input("CNIC (13 digits)", key="v_cnic")
            v_name = st.text_input("Full Name", key="v_name")
            v_roll = st.text_input("University Roll Number", key="v_roll")
            v_society_name = st.selectbox("Select Voter's Society", list(society_options.keys()), key="v_society")

            v_image = render_auto_face_capture("voter")

            if st.button("Register Voter", use_container_width=True):
                if v_image is None:
                    st.warning("Please position your face in front of the camera so the photo can be captured.")
                else:
                    ok, msg = register_voter(v_cnic, v_name, v_roll, society_options[v_society_name], v_image)
                    (st.success if ok else st.warning)(msg)


def render_polling_officer_page():
    st.markdown("### Polling Officer")
    st.caption("Voter registration, vote casting, and turnout/log monitoring - all Polling Officer duties in one place.")
    tab_register, tab_vote, tab_monitor = st.tabs(
        ["Voter Registration", "Cast Your Vote", "Election Officer"])
    with tab_register:
        _render_voter_registration_section()
    with tab_vote:
        render_vote_page()
    with tab_monitor:
        render_election_officer_page()

# -------------------- PAGE: VOTE --------------------

def render_vote_page():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.container(border=True):
            st.markdown("### Voter Login")
            st.caption("Enter your CNIC, then verify your face below")
            vote_cnic = st.text_input("CNIC (13 digits)", key="vote_cnic")

    st.subheader("Step 1: Face Authentication + Liveness Check")
    st.caption("Look straight at the camera with eyes open, then blink naturally. "
               "Both frames are captured automatically.")

    webrtc_ctx = webrtc_streamer(
        key="vote-auto-capture",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AutoCaptureProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        with webrtc_ctx.video_processor.lock:
            proc_state = webrtc_ctx.video_processor.state
            if proc_state == "done" and st.session_state.get("auto_frames_ready") != True:
                st.session_state["auto_frame_open"] = webrtc_ctx.video_processor.frame_open.copy()
                st.session_state["auto_frame_blink"] = webrtc_ctx.video_processor.frame_blink.copy()
                st.session_state["auto_frames_ready"] = True

    if st.session_state.get("auto_frames_ready"):
        st.success("Both frames captured automatically.")
        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            st.image(cv2.cvtColor(st.session_state["auto_frame_open"], cv2.COLOR_BGR2RGB),
                      caption="Captured: eyes open", width="stretch")
        with prev_col2:
            st.image(cv2.cvtColor(st.session_state["auto_frame_blink"], cv2.COLOR_BGR2RGB),
                      caption="Captured: blink", width="stretch")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Verify Face & Liveness"):
                ok, msg, voter_state = verify_voter_face(
                    vote_cnic, st.session_state["auto_frame_open"], st.session_state["auto_frame_blink"])
                st.session_state["face_ok"] = ok
                st.session_state["voter_state"] = voter_state
                # FIX: store the result so the message does not vanish on the next rerun
                st.session_state["verify_msg"] = (ok, msg)
        with col_b:
            if st.button("Retake"):
                # FIX: reset the processor too, otherwise it stays "done" and re-captures the old frames
                if webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.reset()
                for k in ("auto_frame_open", "auto_frame_blink", "auto_frames_ready", "verify_msg"):
                    st.session_state.pop(k, None)
                st.session_state["face_ok"] = False
                st.session_state["voter_state"] = None
                st.rerun()

        # FIX: always show the last verification result
        if st.session_state.get("verify_msg"):
            v_ok, v_msg = st.session_state["verify_msg"]
            (st.success if v_ok else st.error)(v_msg)
    else:
        st.info("Waiting for camera: keep your face centered and eyes open, then blink.")

    st.subheader("Step 2: Location (GPS Radius) Verification")
    st.caption(f"Allowed voting area center: {ALLOWED_CENTER}, radius {ALLOWED_RADIUS_METERS}m")
    col3, col4 = st.columns(2)
    with col3:
        lat = st.number_input("Latitude", value=ALLOWED_CENTER[0], format="%.6f", key="lat_input")
    with col4:
        lon = st.number_input("Longitude", value=ALLOWED_CENTER[1], format="%.6f", key="lon_input")

    if st.button("Verify Location"):
        if not st.session_state["face_ok"] or not st.session_state["voter_state"]:
            st.warning("Please complete Face Verification first.")
        else:
            ok, msg = verify_location(st.session_state["voter_state"], lat, lon)
            st.session_state["gps_ok"] = ok
            (st.success if ok else st.error)(msg)

    st.subheader("Step 3: Select Your Society's Candidate")
    voter_state = st.session_state.get("voter_state")
    if not voter_state:
        st.info("Please complete Face Verification first so your society can be identified.")
        return

    st.caption(f"Your Society: **{voter_state.get('society_name') or 'N/A'}**")
    candidates_df = get_approved_candidates_df(voter_state.get("society_id"))
    st.dataframe(candidates_df, width="stretch")
    if not candidates_df.empty:
        options = {f"{row.ID} - {row.Candidate} (Roll: {row.RollNumber})": row.ID for row in candidates_df.itertuples()}
        choice_label = st.selectbox("Select Candidate to Vote", list(options.keys()), key="candidate_choice")
        if st.button("Cast Vote"):
            if not (st.session_state["face_ok"] and st.session_state["gps_ok"] and st.session_state["voter_state"]):
                st.warning("Please complete Face Verification and Location Verification first.")
            else:
                ok, msg, receipt_code = cast_vote(st.session_state["voter_state"], options[choice_label])
                (st.success if ok else st.error)(msg)
                if ok:
                    st.info(f"**Your vote receipt code: `{receipt_code}`** - save this. You can use it later "
                            f"(under Election Officer > Verify a Receipt) to confirm your vote was recorded, "
                            f"without revealing who you voted for.")
                    st.session_state["face_ok"] = False
                    st.session_state["gps_ok"] = False
                    st.session_state["voter_state"] = None
                    for k in ("auto_frame_open", "auto_frame_blink", "auto_frames_ready", "verify_msg"):
                        st.session_state.pop(k, None)
                    if webrtc_ctx.video_processor:
                        webrtc_ctx.video_processor.reset()
    else:
        st.info("No approved candidate is available for your society yet.")

# -------------------- PAGE: ELECTION OFFICER --------------------

def render_election_officer_page():
    st.subheader("Election Monitoring (View-Only)")
    st.caption("Election Officers can monitor turnout, results, and authentication logs, but cannot modify any data.")
    st.info(get_turnout_stats())
    st.markdown("#### Live Results")
    st.dataframe(get_results_df(), width="stretch")
    st.markdown("#### Authentication Logs")
    st.dataframe(get_auth_logs(), width="stretch")

    st.markdown("#### Verify Election Integrity")
    st.caption("Re-checks the tamper-evident hash chain over every cast vote (NFR-7 Data Integrity). "
               "If any vote row were ever edited, deleted, or reordered directly in the database, "
               "this check would detect exactly where the chain breaks.")
    if st.button("Run Integrity Check", key="eo_verify_chain_btn"):
        is_valid, total, broken_id = verify_vote_chain()
        if total == 0:
            st.info("No votes have been cast yet - nothing to verify.")
        elif is_valid:
            st.success(f"Chain intact - all {total} vote(s) verified. No tampering detected.")
        else:
            st.error(f"Tamper detected! Chain integrity broken at Vote ID {broken_id} "
                      f"(checked {total} vote(s) total).")

    st.markdown("#### Verify a Voter's Receipt")
    st.caption("A voter can bring their receipt code here to confirm their vote was recorded - "
               "without revealing which candidate they chose (ballot secrecy preserved).")
    receipt_col1, receipt_col2 = st.columns([2, 1])
    with receipt_col1:
        receipt_input = st.text_input("Receipt code", key="eo_receipt_input", max_chars=10)
    with receipt_col2:
        st.write("")
        if st.button("Verify Receipt", key="eo_verify_receipt_btn", use_container_width=True):
            ok, msg = verify_receipt_code(receipt_input)
            (st.success if ok else st.warning)(msg)

# -------------------- PAGE: RESULTS --------------------

def render_results_page():
    st.subheader("Real-Time Election Results")
    results_df = get_results_df()
    st.dataframe(results_df, width="stretch")
    if not results_df.empty:
        st.markdown("#### Society-wise Winner")
        st.dataframe(get_society_winners_df(), width="stretch")
        st.bar_chart(results_df.set_index("Candidate")["Votes"])
    else:
        st.info("No results yet.")

# -------------------- ROUTER --------------------

page = st.session_state["page"]

if page != "Home":
    if st.button("Home", key="back_home_btn"):
        st.session_state["page"] = "Home"
        st.rerun()

if page == "Home":
    render_home()
elif page == "Admin":
    render_admin_hub_page()
elif page == "Candidate Registration":
    render_candidate_page()
elif page == "Polling Officer":
    render_polling_officer_page()
elif page == "Results":
    render_results_page()
