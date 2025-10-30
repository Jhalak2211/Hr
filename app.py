# from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_from_directory
# from flask_cors import CORS
# import sqlite3
# from werkzeug.security import generate_password_hash, check_password_hash
# import uuid, os, base64, smtplib, io
# from werkzeug.utils import secure_filename
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import analyzer  # your resume AI analysis module
# from dotenv import load_dotenv

# # -------------------------------
# # CONFIGURATION
# # -------------------------------
# load_dotenv()

# app = Flask(__name__)
# CORS(app)
# app.secret_key = os.environ.get("SECRET_KEY", "super_secret_dev_key")

# DATABASE = "hr_users.db"
# UPLOAD_FOLDER = "uploads"
# ALLOWED_EXTENSIONS = {"pdf", "docx", "png", "jpg", "jpeg"}
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# # -------------------------------
# # HELPERS
# # -------------------------------
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# def get_db_connection():
#     conn = sqlite3.connect(DATABASE)
#     conn.row_factory = sqlite3.Row
#     return conn


# # -------------------------------
# # PAGE ROUTES
# # -------------------------------
# @app.route("/")
# def register_page():
#     return render_template("register.html")


# @app.route("/login")
# def login_page():
#     return render_template("index.html")


# @app.route("/dashboard")
# def dashboard():
#     if "user_id" not in session:
#         return redirect(url_for("login_page"))
#     conn = get_db_connection()
#     jobs = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
#     conn.close()
#     return render_template("dashboard.html", username=session.get("username"), jobs=jobs)


# @app.route("/logout")
# def logout():
#     session.clear()
#     return redirect(url_for("login_page"))


# @app.route("/create-job", methods=["GET"])
# def create_job_page():
#     if "user_id" not in session:
#         return redirect(url_for("login_page"))
#     return render_template("create_job.html")


# @app.route("/apply/<link_id>", methods=["GET"])
# def apply_page(link_id):
#     conn = get_db_connection()
#     job = conn.execute("SELECT * FROM jobs WHERE unique_link_id = ?", (link_id,)).fetchone()
#     conn.close()
#     if job is None:
#         return "Job not found", 404
#     return render_template("apply.html", job=job)


# @app.route("/applicant/<int:applicant_id>")
# def applicant_details(applicant_id):
#     if "user_id" not in session:
#         return redirect(url_for("login_page"))
#     conn = get_db_connection()
#     applicant = conn.execute("SELECT * FROM applications WHERE id = ?", (applicant_id,)).fetchone()
#     conn.close()
#     if applicant is None:
#         return "Applicant not found", 404
#     return render_template("applicant_details.html", applicant=applicant)


# @app.route("/job/<int:job_id>/applicants")
# def view_applicants(job_id):
#     if "user_id" not in session:
#         return redirect(url_for("login_page"))
#     conn = get_db_connection()
#     job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
#     applicants = conn.execute(
#         "SELECT * FROM applications WHERE job_id = ? AND status != 'Discarded' ORDER BY applied_at DESC",
#         (job_id,),
#     ).fetchall()
#     conn.close()
#     if job is None:
#         return "Job not found", 404
#     return render_template("view_applicants.html", job=job, applicants=applicants)


# @app.route("/shortlisted")
# def shortlisted_page():
#     if "user_id" not in session:
#         return redirect(url_for("login_page"))
#     conn = get_db_connection()
#     candidates = conn.execute(
#         """
#         SELECT a.*, j.job_title 
#         FROM applications a 
#         JOIN jobs j ON a.job_id = j.id 
#         WHERE a.status = 'Shortlisted'
#         ORDER BY a.applied_at DESC
#         """
#     ).fetchall()
#     conn.close()
#     return render_template("shortlisted.html", candidates=candidates)


# # -------------------------------
# # AUTHENTICATION
# # -------------------------------
# @app.route("/register", methods=["POST"])
# def register():
#     data = request.get_json()
#     username, password = data.get("username"), data.get("password")
#     if not username or not password:
#         return jsonify({"message": "Username and password are required!"}), 400

#     hashed_password = generate_password_hash(password)
#     conn = get_db_connection()
#     try:
#         conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
#         conn.commit()
#     except sqlite3.IntegrityError:
#         return jsonify({"message": "Username already exists!"}), 400
#     finally:
#         conn.close()
#     return jsonify({"message": "User created successfully!"}), 201


# @app.route("/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     username, password = data.get("username"), data.get("password")
#     if not username or not password:
#         return jsonify({"message": "Invalid credentials!"}), 400

#     conn = get_db_connection()
#     user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
#     conn.close()

#     if user and check_password_hash(user["password"], password):
#         session["user_id"], session["username"] = user["id"], user["username"]
#         return jsonify({"message": "Login successful!"}), 200
#     else:
#         return jsonify({"message": "Invalid credentials!"}), 401


# # -------------------------------
# # CREATE JOB
# # -------------------------------
# @app.route("/create-job", methods=["POST"])
# def create_job():
#     if "user_id" not in session:
#         return jsonify({"message": "Unauthorized"}), 401

#     data = request.get_json()
#     if not data.get("job_title") or not data.get("job_description"):
#         return jsonify({"message": "Job title and description are required"}), 400

#     link_id = str(uuid.uuid4())
#     conn = get_db_connection()
#     conn.execute(
#         """
#         INSERT INTO jobs (job_title, job_description, location, required_skills, resume_keywords, unique_link_id, created_by_user_id)
#         VALUES (?, ?, ?, ?, ?, ?, ?)
#         """,
#         (
#             data.get("job_title"),
#             data.get("job_description"),
#             data.get("location"),
#             data.get("required_skills"),
#             data.get("resume_keywords"),
#             link_id,
#             session["user_id"],
#         ),
#     )
#     conn.commit()
#     conn.close()
#     full_link = url_for("apply_page", link_id=link_id, _external=True)
#     return jsonify({"message": "Job link created!", "link": full_link}), 201


# # -------------------------------
# # APPLY / SCORING + XAI FEEDBACK
# # -------------------------------
# @app.route("/apply/<link_id>", methods=["POST"])
# def handle_application(link_id):
#     conn = get_db_connection()
#     job = conn.execute("SELECT * FROM jobs WHERE unique_link_id = ?", (link_id,)).fetchone()
#     if job is None:
#         conn.close()
#         return jsonify({"message": "Job not found"}), 404

#     if not request.form.get("applicant_name") or not request.form.get("applicant_email"):
#         conn.close()
#         return jsonify({"message": "Name and email are required"}), 400

#     if "resume" not in request.files:
#         conn.close()
#         return jsonify({"message": "Resume file is required"}), 400

#     resume_file = request.files["resume"]
#     if not allowed_file(resume_file.filename):
#         conn.close()
#         return jsonify({"message": "Invalid resume format"}), 400

#     resume_filename = f"{uuid.uuid4()}_{secure_filename(resume_file.filename)}"
#     resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_filename)
#     resume_file.save(resume_path)

#     live_photo_filename = None
#     live_photo_data = request.form.get("live_photo_data", "")
#     if live_photo_data:
#         try:
#             if "," in live_photo_data:
#                 _, encoded = live_photo_data.split(",", 1)
#             else:
#                 encoded = live_photo_data
#             image_data = base64.b64decode(encoded)
#             live_photo_filename = f"{uuid.uuid4()}_live.jpg"
#             with open(os.path.join(app.config["UPLOAD_FOLDER"], live_photo_filename), "wb") as f:
#                 f.write(image_data)
#         except Exception as e:
#             print(f"⚠️ Live photo error: {e}")

#     # Perform Resume Analysis with Explainability
#     match_data = {"score": 0, "matches": [], "misses": []}
#     ai_feedback = "Could not generate AI explanation. Resume or JD may be incomplete."

#     try:
#         with open(resume_path, "rb") as f:
#             resume_bytes = f.read()
#         resume_text = analyzer.parser.extract_text(resume_filename, resume_bytes)
#         jd_text = f"{job['job_description']} {job['required_skills'] or ''}"

#         if resume_text.strip() and jd_text.strip():
#             resume_keywords = analyzer.parser.extract_keywords(resume_text)
#             match_data = analyzer.calculate_match(resume_keywords, jd_text)

#             # ✅ Explainable AI Feedback
#             ai_feedback = analyzer.get_ats_feedback(
#                 resume_text, jd_text,
#                 match_data.get("matches", []),
#                 match_data.get("misses", []),
#                 match_data.get("score", 0)
#             )
#     except Exception as e:
#         print(f"⚠️ Analyzer error: {e}")

#     # Optional Uploaded Photo
#     photo_filename = None
#     photo_file = request.files.get("photo")
#     if photo_file and allowed_file(photo_file.filename):
#         photo_filename = f"{uuid.uuid4()}_{secure_filename(photo_file.filename)}"
#         photo_file.save(os.path.join(app.config["UPLOAD_FOLDER"], photo_filename))

#     # Store in DB
#     conn.execute(
#         """
#         INSERT INTO applications (
#             job_id, applicant_name, applicant_email, applicant_contact,
#             resume_filename, photo_filename, live_photo_filename,
#             match_score, matched_skills, missing_skills, ai_feedback, status
#         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'New')
#         """,
#         (
#             job["id"],
#             request.form["applicant_name"],
#             request.form["applicant_email"],
#             request.form.get("applicant_contact"),
#             resume_filename,
#             photo_filename,
#             live_photo_filename,
#             match_data["score"],
#             ", ".join(match_data.get("matches", [])),
#             ", ".join(match_data.get("misses", [])),
#             ai_feedback,
#         ),
#     )
#     conn.commit()
#     conn.close()

#     return jsonify({"message": "Application submitted successfully!"}), 201


# # -------------------------------
# # THRESHOLD SHORTLIST + EMAIL
# # -------------------------------
# @app.route("/shortlist_above_threshold/<int:job_id>", methods=["POST"])
# def shortlist_above_threshold(job_id):
#     if "user_id" not in session:
#         return jsonify({"message": "Unauthorized"}), 401

#     data = request.get_json()
#     threshold = int(data.get("threshold", 0))

#     conn = get_db_connection()
#     applicants = conn.execute(
#         "SELECT id, applicant_name, applicant_email FROM applications WHERE job_id = ? AND match_score >= ?",
#         (job_id, threshold),
#     ).fetchall()

#     if not applicants:
#         conn.close()
#         return jsonify({"message": f"No applicants above {threshold}%."}), 200

#     sent = 0
#     for a in applicants:
#         if send_shortlist_email(a["applicant_name"], a["applicant_email"]):
#             conn.execute("UPDATE applications SET status = 'Shortlisted' WHERE id = ?", (a["id"],))
#             sent += 1

#     conn.commit()
#     conn.close()

#     return jsonify({"message": f"✅ Sent shortlist emails to {sent} applicants scoring ≥ {threshold}%."}), 200


# def send_shortlist_email(name, email):
#     try:
#         sender = os.environ.get("SMTP_EMAIL")
#         password = os.environ.get("SMTP_PASS")
#         smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
#         smtp_port = int(os.environ.get("SMTP_PORT", 587))

#         subject = "🎉 You Have Been Shortlisted!"
#         body = f"""
#         Dear {name},

#         Congratulations! Based on your resume evaluation, you have been shortlisted for the next stage.

#         Our HR team will reach out with further steps soon.

#         Regards,
#         Talent Acquisition Team
#         """

#         msg = MIMEMultipart()
#         msg["From"], msg["To"], msg["Subject"] = sender, email, subject
#         msg.attach(MIMEText(body, "plain"))

#         with smtplib.SMTP(smtp_server, smtp_port) as server:
#             server.starttls()
#             server.login(sender, password)
#             server.send_message(msg)

#         print(f"✅ Email sent to {email}")
#         return True
#     except Exception as e:
#         print(f"⚠️ Email error ({email}): {e}")
#         return False


# # -------------------------------
# # FILE SERVING
# # -------------------------------
# @app.route("/uploads/<path:filename>")
# def uploaded_file(filename):
#     if "user_id" not in session:
#         return "Unauthorized", 401
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# # -------------------------------
# # MAIN
# # -------------------------------
# if __name__ == "__main__":
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(host="0.0.0.0", port=5000, debug=True)












# ============================================================
# ⚡ FAST MODEL STARTUP OPTIMIZATION (REAL MODELS, NO DUMMIES)
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # silence TensorFlow logs
os.environ["TFHUB_CACHE_DIR"] = r"C:\Models\tfhub"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"C:\Models\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"C:\Models\transformers"
os.environ["TFDS_DATA_DIR"] = r"C:\Models\tfds"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ⚙️ Create folders if missing
for folder in [
    r"C:\Models\tfhub",
    r"C:\Models\huggingface",
    r"C:\Models\transformers",
    r"C:\Models\tfds"
]:
    os.makedirs(folder, exist_ok=True)

# 🧩 TensorFlow eager disable for faster load + inference
try:
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
except Exception as e:
    print("⚠️ TensorFlow not yet loaded:", e)

# ⚙️ Force CPU mode for stability (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



from flask import Flask, request, jsonify, render_template, render_template_string, session, redirect, url_for, send_from_directory, abort
from flask_cors import CORS
import sqlite3, uuid, os, base64, smtplib, io, tempfile, datetime, json
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os, uuid, sqlite3, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import spacy
# ⚡ Use prelinked + partial spaCy pipeline
import spacy
try:
    nlp = spacy.load("en", disable=["parser", "ner", "lemmatizer"])
except:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
print("✅ spaCy loaded quickly (light mode)")

# ============================================================
# 🚀 Lightweight AI Model Loader (Optimized for Fast Startup)
# ============================================================

import os, threading
from functools import lru_cache

# -----------------------------------
# ✅ TensorFlow CPU-Only (Safe & Fast)
# -----------------------------------
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logs
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    print("⚠️ TensorFlow not available, skipping:", e)
    tf = None
    load_model = None

# -----------------------------------
# ✅ Lazy-load Models On-Demand
# -----------------------------------

_model_lock = threading.Lock()
_conf_model = None
_sentence_model = None
_nlp = None

@lru_cache(maxsize=1)
def get_conf_model():
    """Load confidence TensorFlow model only once (CPU optimized)."""
    global _conf_model
    with _model_lock:
        if _conf_model is None and load_model is not None:
            try:
                print("⚙️ Loading confidence model on-demand...")
                _conf_model = tf.keras.models.load_model("best_confidence_model_tf")
            except Exception as e:
                print("⚠️ Confidence model not found — mock mode:", e)
                _conf_model = None
    return _conf_model


@lru_cache(maxsize=1)
def get_sentence_model():
    """Load SentenceTransformer model only once (for relevance/embedding)."""
    global _sentence_model
    with _model_lock:
        if _sentence_model is None:
            try:
                print("⚙️ Loading SentenceTransformer (MiniLM)...")
                from sentence_transformers import SentenceTransformer
                _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print("⚠️ Failed to load SentenceTransformer — mock mode:", e)
                _sentence_model = None
    return _sentence_model


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy model only once for NLP processing."""
    global _nlp
    with _model_lock:
        if _nlp is None:
            try:
                print("⚙️ Loading spaCy model on-demand...")
                import spacy
                _nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print("⚠️ spaCy model not found — mock mode:", e)
                _nlp = None
    return _nlp


# -----------------------------------
# ✅ Background Pre-Warming (optional)
# -----------------------------------
def prewarm_models():
    """Warm up models in background so the first candidate doesn't wait."""
    try:
        threading.Thread(target=get_conf_model, daemon=True).start()
        threading.Thread(target=get_sentence_model, daemon=True).start()
        threading.Thread(target=get_nlp, daemon=True).start()
    except Exception as e:
        print("⚠️ Model prewarm failed:", e)

# Start prewarm automatically when app loads
prewarm_models()


# -----------------------------------
# 🧠 Memory-Safe Auto-Unload
# -----------------------------------
import time

_model_last_used = {}

def _mark_used(name: str):
    """Mark a model as recently used."""
    _model_last_used[name] = time.time()

def _gc_models():
    """Unload models that have been idle > 1800 s (30 min)."""
    now = time.time()
    with _model_lock:
        global _conf_model, _sentence_model, _nlp
        for name, model_ref in [
            ("conf", _conf_model),
            ("sent", _sentence_model),
            ("nlp", _nlp)
        ]:
            last = _model_last_used.get(name, now)
            if model_ref is not None and (now - last) > 1800:
                print(f"🧹 Unloading idle model: {name}")
                if name == "conf": _conf_model = None
                if name == "sent": _sentence_model = None
                if name == "nlp": _nlp = None

def _start_model_gc():
    """Background thread that checks every 10 min."""
    def loop():
        while True:
            time.sleep(600)
            _gc_models()
    threading.Thread(target=loop, daemon=True).start()

_start_model_gc()







# ============================================================
# ✅ Usage Example (inside your routes later)
# ============================================================
# model = get_conf_model()
# if model:
#     confidence = float(model.predict(x)[0][0])
# else:
#     confidence = 0.7  # mock fallback
#
# emb_model = get_sentence_model()
# nlp = get_nlp()


# import the Gemini app as-is
from combined_app import app as candidate_app  # uses its own DB (test_system.db)


# === Optional/soft deps for test features ===
try:
    import face_recognition
    FACE_OK = True
except Exception:
    FACE_OK = False

try:
    import speech_recognition as sr
    SR_OK = True
except Exception:
    SR_OK = False

import analyzer  # your resume & JD analyzer (with Gemini + OCR + XAI fallback)
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------------------
# Flask setup
# ------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_dev_key")

DATABASE = "hr_users.db"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "png", "jpg", "jpeg", "webm", "wav", "mp3", "m4a"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def send_email(to_email, subject, body):
    try:
        sender = os.environ.get("SMTP_EMAIL")
        password = os.environ.get("SMTP_PASS")
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = sender, to_email, subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print(f"✅ Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"⚠️ Email error ({to_email}): {e}")
        return False

# ------------------------------------------------------------------------------
# Core pages already in your app
# ------------------------------------------------------------------------------
@app.route("/")
def register_page():
    return render_template("register.html")

@app.route("/login")
def login_page():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    db = get_db()
    jobs = db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    db.close()
    return render_template("dashboard.html", username=session.get("username"), jobs=jobs)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/create-job", methods=["GET"])
def create_job_page():
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    return render_template("create_job.html")

@app.route("/apply/<link_id>", methods=["GET"])
def apply_page(link_id):
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE unique_link_id = ?", (link_id,)).fetchone()
    db.close()
    if job is None:
        return "Job not found", 404
    return render_template("apply.html", job=job)

@app.route("/applicant/<int:applicant_id>")
def applicant_details(applicant_id):
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    db = get_db()
    applicant = db.execute("SELECT * FROM applications WHERE id = ?", (applicant_id,)).fetchone()
    db.close()
    if applicant is None:
        return "Applicant not found", 404
    return render_template("applicant_details.html", applicant=applicant)

@app.route("/job/<int:job_id>/applicants")
def view_applicants(job_id):
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    applicants = db.execute(
        "SELECT * FROM applications WHERE job_id = ? AND status != 'Discarded' ORDER BY applied_at DESC",
        (job_id,),
    ).fetchall()
    db.close()
    if job is None:
        return "Job not found", 404
    return render_template("view_applicants.html", job=job, applicants=applicants)

@app.route("/shortlisted")
def shortlisted_page():
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    db = get_db()
    candidates = db.execute("""
        SELECT a.*, j.job_title
        FROM applications a
        JOIN jobs j ON a.job_id = j.id
        WHERE a.status = 'Shortlisted'
        ORDER BY a.applied_at DESC
    """).fetchall()
    db.close()
    return render_template("shortlisted.html", candidates=candidates)

# ------------------------------------------------------------------------------
# Auth & Job creation APIs
# ------------------------------------------------------------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username, password = data.get("username"), data.get("password")
    if not username or not password:
        return jsonify({"message": "Username and password are required!"}), 400
    hashed_password = generate_password_hash(password)
    db = get_db()
    try:
        db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"message": "Username already exists!"}), 400
    finally:
        db.close()
    return jsonify({"message": "User created successfully!"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username, password = data.get("username"), data.get("password")
    if not username or not password:
        return jsonify({"message": "Invalid credentials!"}), 400
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    db.close()
    if user and check_password_hash(user["password"], password):
        session["user_id"], session["username"] = user["id"], user["username"]
        return jsonify({"message": "Login successful!"}), 200
    return jsonify({"message": "Invalid credentials!"}), 401

@app.route("/create-job", methods=["POST"])
def create_job():
    if "user_id" not in session:
        return jsonify({"message": "Unauthorized"}), 401
    data = request.get_json()
    if not data.get("job_title") or not data.get("job_description"):
        return jsonify({"message": "Job title and description are required"}), 400
    link_id = str(uuid.uuid4())
    db = get_db()
    db.execute("""
        INSERT INTO jobs (job_title, job_description, location, required_skills, resume_keywords, unique_link_id, created_by_user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (data.get("job_title"), data.get("job_description"), data.get("location"),
          data.get("required_skills"), data.get("resume_keywords"), link_id, session["user_id"]))
    db.commit()
    db.close()
    full_link = url_for("apply_page", link_id=link_id, _external=True)
    return jsonify({"message": "Job link created!", "link": full_link}), 201

# ------------------------------------------------------------------------------
# Application intake (resume scoring + XAI)
# ------------------------------------------------------------------------------
@app.route("/apply/<link_id>", methods=["POST"])
def handle_application(link_id):
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE unique_link_id = ?", (link_id,)).fetchone()
    if job is None:
        db.close()
        return jsonify({"message": "Job not found"}), 404

    if not request.form.get("applicant_name") or not request.form.get("applicant_email"):
        db.close()
        return jsonify({"message": "Name and email are required"}), 400

    if "resume" not in request.files:
        db.close()
        return jsonify({"message": "Resume file is required"}), 400

    resume_file = request.files["resume"]
    if not allowed_file(resume_file.filename):
        db.close()
        return jsonify({"message": "Invalid resume format"}), 400

    resume_filename = f"{uuid.uuid4()}_{secure_filename(resume_file.filename)}"
    resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_filename)
    resume_file.save(resume_path)

    # Live photo from camera (optional)
    live_photo_filename = None
    live_photo_data = request.form.get("live_photo_data", "")
    if live_photo_data:
        try:
            encoded = live_photo_data.split(",", 1)[1] if "," in live_photo_data else live_photo_data
            image_data = base64.b64decode(encoded)
            live_photo_filename = f"{uuid.uuid4()}_live.jpg"
            with open(os.path.join(app.config["UPLOAD_FOLDER"], live_photo_filename), "wb") as f:
                f.write(image_data)
        except Exception as e:
            print(f"⚠️ Live photo error: {e}")

    # Analysis
    match_data = {"score": 0, "matches": [], "misses": []}
    ai_feedback = "Could not generate AI explanation. Resume or JD may be incomplete."
    try:
        with open(resume_path, "rb") as f:
            resume_bytes = f.read()
        resume_text = analyzer.parser.extract_text(resume_filename, resume_bytes)
        jd_text = f"{job['job_description']} {job['required_skills'] or ''}"
        if resume_text.strip() and jd_text.strip():
            resume_keywords = analyzer.parser.extract_keywords(resume_text)
            match_data = analyzer.calculate_match(resume_keywords, jd_text)
            ai_feedback = analyzer.get_ats_feedback(
                resume_text, jd_text,
                match_data.get("matches", []),
                match_data.get("misses", []),
                match_data.get("score", 0)
            )
    except Exception as e:
        print(f"⚠️ Analyzer error: {e}")

    # Optional static photo
    photo_filename = None
    photo_file = request.files.get("photo")
    if photo_file and allowed_file(photo_file.filename):
        photo_filename = f"{uuid.uuid4()}_{secure_filename(photo_file.filename)}"
        photo_file.save(os.path.join(app.config["UPLOAD_FOLDER"], photo_filename))

    # Save application
    db.execute("""
        INSERT INTO applications (
            job_id, applicant_name, applicant_email, applicant_contact,
            resume_filename, photo_filename, live_photo_filename,
            match_score, matched_skills, missing_skills, ai_feedback, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'New')
    """, (job["id"], request.form["applicant_name"], request.form["applicant_email"],
          request.form.get("applicant_contact"), resume_filename, photo_filename, live_photo_filename,
          match_data["score"], ", ".join(match_data.get("matches", [])),
          ", ".join(match_data.get("misses", [])), ai_feedback))
    db.commit()
    db.close()
    return jsonify({"message": "Application submitted successfully!"}), 201

# ------------------------------------------------------------------------------
# Shortlist threshold emailing (existing)
# ------------------------------------------------------------------------------
@app.route("/shortlist_above_threshold/<int:job_id>", methods=["POST"])
def shortlist_above_threshold(job_id):
    if "user_id" not in session:
        return jsonify({"message": "Unauthorized"}), 401
    data = request.get_json()
    threshold = int(data.get("threshold", 0))
    db = get_db()
    applicants = db.execute(
        "SELECT id, applicant_name, applicant_email FROM applications WHERE job_id = ? AND match_score >= ?",
        (job_id, threshold)
    ).fetchall()
    if not applicants:
        db.close()
        return jsonify({"message": f"No applicants above {threshold}%."}), 200
    sent = 0
    for a in applicants:
        if send_email(a["applicant_email"], "You Have Been Shortlisted!", 
                      f"Dear {a['applicant_name']},\n\nYou have been shortlisted for the next round.\n\nRegards,\nTalent Team"):
            db.execute("UPDATE applications SET status = 'Shortlisted' WHERE id = ?", (a["id"],))
            sent += 1
    db.commit()
    db.close()
    return jsonify({"message": f"✅ Sent shortlist emails to {sent} applicants scoring ≥ {threshold}%."}), 200

# ------------------------------------------------------------------------------
# === TESTING MODULE (Gemini generation, per-page, audio, face-check) ==========
# ------------------------------------------------------------------------------

# Admin page to configure & generate a test for a job
@app.route("/tests/create/<int:job_id>", methods=["GET"])
def tests_create_page(job_id):
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    db.close()
    if not job:
        return "Job not found", 404
    # A tiny inline page (you can move to templates/tests_create.html)
    return render_template_string("""
<!doctype html><html><head>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
<title>Create Test</title></head><body class="p-4">
<h3>Create Test for: {{ job['job_title'] }}</h3><hr>
<form id="cfg" class="mb-3">
  <div class="form-row">
    <div class="col">
      <label>Logical Reasoning</label>
      <input class="form-control" id="logical" type="number" min="0" value="3">
    </div>
    <div class="col">
      <label>Cognitive Ability</label>
      <input class="form-control" id="cognitive" type="number" min="0" value="0">
    </div>
    <div class="col">
      <label>Scenario-Based</label>
      <input class="form-control" id="scenario" type="number" min="0" value="3">
    </div>
    <div class="col">
      <label>Technical</label>
      <input class="form-control" id="technical" type="number" min="0" value="0">
    </div>
  </div>
  <div class="form-group mt-3">
    <label>Technical Topics (comma separated)</label>
    <input class="form-control" id="topics" placeholder="e.g., Python, Flask, SQL">
  </div>
  <div class="form-row">
    <div class="col">
      <label>Total Duration (minutes)</label>
      <input class="form-control" id="duration" type="number" min="5" value="25">
    </div>
  </div>
  <button type="button" id="btnGen" class="btn btn-primary mt-3">Generate & Save Test</button>
  <button type="button" id="btnSend" class="btn btn-success mt-3 ml-2">Generate & Send Links to Shortlisted</button>
</form>
<div id="msg"></div>
<script>
async function gen(send){
  const payload = {
    logical: Number(document.getElementById('logical').value||0),
    cognitive: Number(document.getElementById('cognitive').value||0),
    scenario: Number(document.getElementById('scenario').value||0),
    technical: Number(document.getElementById('technical').value||0),
    topics: document.getElementById('topics').value||'',
    duration: Number(document.getElementById('duration').value||20),
    send_links: !!send
  };
  const res = await fetch('{{ url_for("tests_generate", job_id=job["id"]) }}', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  const out = await res.json();
  document.getElementById('msg').innerText = out.message || JSON.stringify(out);
}
document.getElementById('btnGen').onclick=()=>gen(false);
document.getElementById('btnSend').onclick=()=>gen(true);
</script>
</body></html>
""", job=job)

# Generate test (questions via Gemini) and optionally send links
@app.route("/tests/generate/<int:job_id>", methods=["POST"])
def tests_generate(job_id):
    if "user_id" not in session:
        return jsonify({"message":"Unauthorized"}), 401
    data = request.get_json() or {}
    n_logical = int(data.get("logical", 0))
    n_cognitive = int(data.get("cognitive", 0))
    n_scenario = int(data.get("scenario", 0))
    n_technical = int(data.get("technical", 0))
    topics = data.get("topics", "")
    duration = int(data.get("duration", 20))
    send_links = bool(data.get("send_links", False))

    # Create test
    db = get_db()
    db.execute("INSERT INTO tests (job_id, total_duration) VALUES (?, ?)", (job_id, duration))
    test_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    # Generate questions using Gemini (with safe fallback)
    qs = generate_questions_with_gemini(n_logical, n_cognitive, n_scenario, n_technical, topics)
    for q in qs:
        db.execute("""
            INSERT INTO test_questions
            (test_id, question_type, question_text, option_a, option_b, option_c, option_d, correct_answer, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (test_id, q["type"], q["text"], q["A"], q["B"], q["C"], q["D"], q["answer"], q["explain"]))
    db.commit()

    if send_links:
        # Send to all shortlisted
        apps = db.execute("SELECT id, applicant_name, applicant_email, live_photo_filename FROM applications WHERE job_id=? AND status='Shortlisted'", (job_id,)).fetchall()
        sent = 0
        for a in apps:
            token = str(uuid.uuid4())
            db.execute("""
                INSERT INTO test_invites (test_id, applicant_id, token, started, completed)
                VALUES (?, ?, ?, 0, 0)
            """, (test_id, a["id"], token))
            link = url_for("take_test", token=token, _external=True)
            if send_email(a["applicant_email"], "Your Assessment Link", f"Dear {a['applicant_name']},\n\nPlease take your assessment here:\n{link}\n\nRegards,\nTalent Team"):
                sent += 1
        db.commit()
        db.close()
        return jsonify({"message": f"✅ Test created (ID {test_id}) and links sent to {sent} shortlisted candidates."}), 201

    db.close()
    return jsonify({"message": f"✅ Test created (ID {test_id}) with {len(qs)} questions. Use 'Generate & Send' to email links."}), 201

def generate_questions_with_gemini(n_logical, n_cognitive, n_scenario, n_technical, topics):
    """
    Use analyzer.llm (Gemini) if available; otherwise return a basic fallback set.
    Each question has: type, text, A/B/C/D, answer, explain.
    """
    total = n_logical + n_cognitive + n_scenario + n_technical
    if total == 0:
        return []

    # If Gemini configured in analyzer, query it
    qs = []
    if getattr(analyzer, "llm", None):
        try:
            prompt = f"""
            Generate interview questions as JSON array. Each item fields:
            type (Logical|Cognitive|Scenario|Technical),
            text, A, B, C, D, answer (A|B|C|D), explain (short).
            Counts: Logical={n_logical}, Cognitive={n_cognitive}, Scenario={n_scenario}, Technical={n_technical}.
            Technical topics: {topics or "general software engineering"}.
            Keep them crisp, one-concept each, difficulty moderate.
            """
            resp = analyzer.llm.generate_content(prompt)
            # Try parse JSON; if model returns markdown, strip code fences
            raw = resp.text.strip()
            raw = raw.strip("`")
            start = raw.find("[")
            end = raw.rfind("]")
            raw_json = raw[start:end+1] if start != -1 and end != -1 else "[]"
            qs = json.loads(raw_json)
        except Exception as e:
            print(f"⚠️ Gemini generation failed, using fallback: {e}")

    if not qs:
        # Simple fallback bank
        if n_logical:
            qs.append({"type":"Logical","text":"If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?","A":"Yes","B":"No","C":"Only sometimes","D":"Not enough info","answer":"A","explain":"Transitive property."})
        if n_cognitive:
            qs.append({"type":"Cognitive","text":"What number comes next? 2, 4, 8, 16, ?","A":"24","B":"30","C":"32","D":"36","answer":"C","explain":"Doubles each time."})
        if n_scenario:
            qs.append({"type":"Scenario","text":"You find a production bug at 6pm. What do you do first?","A":"Email the team","B":"Panic","C":"Reproduce & triage impact","D":"Ignore","answer":"C","explain":"Assess impact, reproduce, triage."})
        if n_technical:
            qs.append({"type":"Technical","text":"Which HTTP method is idempotent?","A":"POST","B":"PUT","C":"PATCH","D":"CONNECT","answer":"B","explain":"PUT creates/replaces resource and is idempotent."})
    # Normalize
    norm = []
    for q in qs:
        norm.append({
            "type": q.get("type","Technical"),
            "text": q.get("text",""),
            "A": q.get("A",""),
            "B": q.get("B",""),
            "C": q.get("C",""),
            "D": q.get("D",""),
            "answer": (q.get("answer","A") or "A").strip()[:1].upper(),
            "explain": q.get("explain","")
        })
    return norm

# Candidate entry point (face check screen + start)
@app.route("/take_test/<token>", methods=["GET"])
def take_test(token):
    db = get_db()
    invite = db.execute("""
        SELECT ti.*, a.applicant_name, a.live_photo_filename, t.id AS test_id, t.total_duration
        FROM test_invites ti
        JOIN applications a ON a.id = ti.applicant_id
        JOIN tests t ON t.id = ti.test_id
        WHERE ti.token = ?
    """, (token,)).fetchone()
    db.close()
    if not invite:
        return "Invalid test link.", 404

    # Minimal face-verify start page (inline template for brevity)
    return render_template_string("""
<!doctype html><html><head>
<title>Start Test</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head><body class="p-4">
<h4>Welcome {{ name }}</h4>
<p>Please capture a live photo to verify your identity before starting the test.</p>
<div class="mb-2">
  <video id="video" width="360" height="270" autoplay class="border"></video><br>
  <button id="snap" class="btn btn-primary mt-2">Capture & Verify</button>
</div>
<canvas id="canvas" width="360" height="270" style="display:none;"></canvas>
<div id="msg" class="mt-3 text-info"></div>
<script>
const v = document.getElementById('video');
navigator.mediaDevices.getUserMedia({video:true}).then(s=>v.srcObject=s);
document.getElementById('snap').onclick = async ()=>{
  const c = document.getElementById('canvas'), ctx=c.getContext('2d');
  ctx.drawImage(v,0,0,c.width,c.height);
  const data = c.toDataURL('image/jpeg');
  const res = await fetch('{{ url_for("verify_face", token=token) }}', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({photo:data})
  });
  const out = await res.json();
  if(out.ok){
    location.href='{{ url_for("question_page", token=token, qno=1) }}';
  } else {
    document.getElementById('msg').innerText = out.message;
  }
};
</script>
</body></html>
""", token=token, name=invite["applicant_name"])

@app.route("/verify_face/<token>", methods=["POST"])
def verify_face(token):
    db = get_db()
    invite = db.execute("""
        SELECT ti.*, a.live_photo_filename
        FROM test_invites ti
        JOIN applications a ON a.id = ti.applicant_id
        WHERE ti.token = ?
    """, (token,)).fetchone()
    if not invite:
        db.close()
        return jsonify({"ok": False, "message":"Invalid link"}), 404

    data = request.get_json() or {}
    photo = data.get("photo","")
    if not photo:
        db.close()
        return jsonify({"ok": False, "message":"No image received"}), 400

    # If face_recognition not available, allow pass (to avoid blocking)
    if not FACE_OK:
        db.execute("UPDATE test_invites SET started=1 WHERE token=?", (token,))
        db.commit(); db.close()
        return jsonify({"ok": True, "message":"Face module unavailable, skipping verification."})

    try:
        # Save temp current capture
        encoded = photo.split(",",1)[1] if "," in photo else photo
        b = base64.b64decode(encoded)
        temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_live_now.jpg")
        with open(temp_path,"wb") as f: f.write(b)

        # Compare with stored live photo
        stored = invite["live_photo_filename"]
        stored_path = os.path.join(UPLOAD_FOLDER, stored) if stored else None
        ok = False
        if stored_path and os.path.exists(stored_path):
            known = face_recognition.load_image_file(stored_path)
            unk = face_recognition.load_image_file(temp_path)
            known_enc = face_recognition.face_encodings(known)
            unk_enc = face_recognition.face_encodings(unk)
            if known_enc and unk_enc:
                results = face_recognition.compare_faces([known_enc[0]], unk_enc[0], tolerance=0.55)
                ok = bool(results[0])
        os.remove(temp_path)
        if ok:
            db.execute("UPDATE test_invites SET started=1 WHERE token=?", (token,))
            db.commit(); db.close()
            return jsonify({"ok": True, "message":"Verified"})
        else:
            db.close()
            return jsonify({"ok": False, "message":"Face mismatch. Please contact HR."})
    except Exception as e:
        print("⚠️ Face verify error:", e)
        db.execute("UPDATE test_invites SET started=1 WHERE token=?", (token,))
        db.commit(); db.close()
        return jsonify({"ok": True, "message":"Verification skipped due to error."})

# Per-question page (one question per page + per-test timer client-side)
@app.route("/question/<token>/<int:qno>", methods=["GET"])
def question_page(token, qno):
    db = get_db()
    row = db.execute("""
        SELECT ti.*, t.total_duration, a.applicant_name
        FROM test_invites ti JOIN tests t ON t.id = ti.test_id
        JOIN applications a ON a.id = ti.applicant_id
        WHERE ti.token = ?
    """, (token,)).fetchone()
    if not row:
        db.close(); return "Invalid link.", 404
    qs = db.execute("SELECT * FROM test_questions WHERE test_id=? ORDER BY id", (row["test_id"],)).fetchall()
    if qno < 1 or qno > len(qs):
        db.close(); return redirect(url_for("submit_test", token=token))
    q = qs[qno-1]
    total = len(qs)
    db.close()

    # Scenario questions allow audio recording upload
    is_scenario = (q["question_type"].lower() == "scenario")

    return render_template_string("""
<!doctype html><html><head>
<title>Question {{ qno }} / {{ total }}</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head><body class="p-4">
<div class="d-flex justify-content-between align-items-center">
  <h4>Question {{ qno }} / {{ total }}</h4>
  <div><strong>Time Tips:</strong> Complete within overall duration set by HR.</div>
</div>
<hr>
<div class="mb-3">
  <span class="badge badge-info">{{ q['question_type'] }}</span>
</div>
<p style="font-size:1.1rem">{{ q['question_text'] }}</p>

{% if is_scenario %}
  <div class="alert alert-secondary">
    <strong>Scenario Task:</strong> Please record your answer (max ~60s).
  </div>
  <div class="mb-2">
    <button id="startBtn" class="btn btn-outline-primary btn-sm">Start Recording</button>
    <button id="stopBtn" class="btn btn-outline-danger btn-sm" disabled>Stop</button>
  </div>
  <audio id="playback" controls style="display:none;"></audio>
  <form id="ansForm" class="mt-3">
    <input type="hidden" name="mode" value="audio">
    <input type="hidden" name="blob" id="blob64">
    <button class="btn btn-primary">Submit & Next</button>
  </form>
{% else %}
  <form id="ansForm">
    <input type="hidden" name="mode" value="mcq">
    <div class="list-group">
      <label class="list-group-item"><input type="radio" name="choice" value="A"> A) {{ q['option_a'] }}</label>
      <label class="list-group-item"><input type="radio" name="choice" value="B"> B) {{ q['option_b'] }}</label>
      <label class="list-group-item"><input type="radio" name="choice" value="C"> C) {{ q['option_c'] }}</label>
      <label class="list-group-item"><input type="radio" name="choice" value="D"> D) {{ q['option_d'] }}</label>
    </div>
    <button class="btn btn-primary mt-3">Submit & Next</button>
  </form>
{% endif %}

<script>
const nextUrl = "{{ url_for('answer_question', token=token, qno=qno) }}";
document.getElementById('ansForm').onsubmit = async (e)=>{
  e.preventDefault();
  const fd = new FormData(e.target);
  const res = await fetch(nextUrl, {method:'POST', body:fd});
  const out = await res.json();
  if(out.ok){ location.href = out.next; } else { alert(out.message||'Error'); }
};
{% if is_scenario %}
let mediaRecorder, chunks=[];
const playback = document.getElementById('playback');
document.getElementById('startBtn').onclick = async ()=>{
  const stream = await navigator.mediaDevices.getUserMedia({audio:true});
  mediaRecorder = new MediaRecorder(stream);
  chunks=[];
  mediaRecorder.ondataavailable = e=>chunks.push(e.data);
  mediaRecorder.onstop = e=>{
    const blob = new Blob(chunks, {type:'audio/webm'});
    const reader = new FileReader();
    reader.onloadend = ()=>{
      document.getElementById('blob64').value = reader.result; // data:...base64
      playback.src = reader.result; playback.style.display='block';
    };
    reader.readAsDataURL(blob);
  };
  mediaRecorder.start();
  document.getElementById('startBtn').disabled=true;
  document.getElementById('stopBtn').disabled=false;
};
document.getElementById('stopBtn').onclick = ()=>{
  mediaRecorder.stop();
  document.getElementById('stopBtn').disabled=true;
};
{% endif %}
</script>
</body></html>
""", token=token, q=q, qno=qno, total=total, is_scenario=is_scenario)

# Record answer (MCQ or Audio) and move next
@app.route("/answer/<token>/<int:qno>", methods=["POST"])
def answer_question(token, qno):
    db = get_db()
    invite = db.execute("SELECT * FROM test_invites WHERE token=?", (token,)).fetchone()
    if not invite:
        db.close(); return jsonify({"ok":False,"message":"Invalid link"}), 404
    qrows = db.execute("SELECT * FROM test_questions WHERE test_id=? ORDER BY id", (invite["test_id"],)).fetchall()
    if qno < 1 or qno > len(qrows):
        db.close(); return jsonify({"ok":True,"next": url_for("submit_test", token=token)})
    q = qrows[qno-1]

    mode = request.form.get("mode")
    if mode == "mcq":
        choice = request.form.get("choice","").strip()[:1].upper()
        db.execute("""
            INSERT INTO test_answers (invite_id, question_id, answer_text, is_audio)
            VALUES (?, ?, ?, 0)
        """, (invite["id"], q["id"], choice))
    else:
        # audio blob
        b64 = request.form.get("blob","")
        if not b64:
            db.close(); return jsonify({"ok":False,"message":"No audio provided"}), 400
        encoded = b64.split(",",1)[1] if "," in b64 else b64
        audio_filename = f"{uuid.uuid4()}_ans.webm"
        with open(os.path.join(UPLOAD_FOLDER, audio_filename),"wb") as f:
            f.write(base64.b64decode(encoded))
        db.execute("""
            INSERT INTO test_answers (invite_id, question_id, answer_text, is_audio, audio_filename)
            VALUES (?, ?, ?, 1, ?)
        """, (invite["id"], q["id"], "[AUDIO]", audio_filename))

    db.commit()
    # Next
    next_q = qno + 1
    if next_q > len(qrows):
        db.close()
        return jsonify({"ok":True, "next": url_for("submit_test", token=token)})
    db.close()
    return jsonify({"ok":True, "next": url_for("question_page", token=token, qno=next_q)})

# Submit test: score MCQ locally + Scenario via Gemini (with SR fallback)
@app.route("/submit_test/<token>", methods=["GET","POST"])
def submit_test(token):
    db = get_db()
    invite = db.execute("""
        SELECT ti.*, a.applicant_name, a.applicant_email, t.job_id
        FROM test_invites ti
        JOIN applications a ON a.id = ti.applicant_id
        JOIN tests t ON t.id = ti.test_id
        WHERE ti.token=?
    """,(token,)).fetchone()
    if not invite:
        db.close(); return "Invalid link.", 404

    qs = db.execute("SELECT * FROM test_questions WHERE test_id=? ORDER BY id", (invite["test_id"],)).fetchall()
    ans = db.execute("SELECT * FROM test_answers WHERE invite_id=?",(invite["id"],)).fetchall()
    # Build map question_id -> answer
    amap = {a["question_id"]: a for a in ans}

    total = 0
    mcq_max = 0
    scenario_scores = []
    xai_parts = []

    # Score MCQs
    for q in qs:
        if q["question_type"].lower() != "scenario":
            mcq_max += 1
            arow = amap.get(q["id"])
            chosen = (arow["answer_text"] if arow else "").strip()[:1].upper()
            if chosen and chosen == (q["correct_answer"] or "").strip()[:1].upper():
                total += 1
            expl = q["explanation"] or ""
            xai_parts.append(f"Q: {q['question_text']}\nCorrect: {q['correct_answer']}\nYour: {chosen or '-'}\nWhy: {expl}\n")

    # Scenario: transcribe (if possible) and ask Gemini to score (0-1)
    for q in qs:
        if q["question_type"].lower() == "scenario":
            arow = amap.get(q["id"])
            transcript = ""
            if arow and arow["is_audio"] and arow["audio_filename"]:
                audio_path = os.path.join(UPLOAD_FOLDER, arow["audio_filename"])
                if SR_OK:
                    try:
                        r = sr.Recognizer()
                        with sr.AudioFile(audio_path) as source:
                            audio = r.record(source)
                        transcript = r.recognize_google(audio, show_all=False)
                    except Exception as e:
                        print("⚠️ Speech recognition failed:", e)
                # If no SR, leave transcript empty; Gemini can still grade with a note.
            # Ask Gemini to grade scenario answer (0-100) with explanation
            scen_score, scen_expl = grade_scenario_with_gemini(q["question_text"], transcript)
            scenario_scores.append(scen_score)
            xai_parts.append(f"[Scenario]\nQ: {q['question_text']}\nTranscript: {transcript or '(audio)'}\nScore: {scen_score}\nWhy: {scen_expl}\n")

    # Combine: MCQ contributes proportionally, Scenario avg adds on same scale
    if mcq_max > 0:
        mcq_pct = (total / mcq_max) * 100.0
    else:
        mcq_pct = 0.0
    scen_pct = (sum(scenario_scores)/len(scenario_scores)) if scenario_scores else 0.0
    final_score = int(round((mcq_pct + scen_pct) / (2 if scenario_scores else 1)))

    # Save attempt with XAI feedback
    xai_feedback = "\n".join(xai_parts)
    db.execute("""
        INSERT INTO test_attempts (test_id, applicant_id, total_score, ai_feedback)
        VALUES (?, ?, ?, ?)
    """,(invite["test_id"], invite["applicant_id"], final_score, xai_feedback))
    db.execute("UPDATE test_invites SET completed=1 WHERE id=?", (invite["id"],))
    db.commit(); db.close()

    return render_template_string("""
<!doctype html><html><head>
<title>Test Submitted</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head><body class="p-4">
<h3>Thanks! Your test has been submitted.</h3>
<p><strong>Your Score:</strong> {{ score }}%</p>
<hr>
<h5>Breakdown & AI Explanation</h5>
<pre style="white-space:pre-wrap; background:#f8f9fa; padding:1rem; border-radius:.5rem;">{{ xai }}</pre>
</body></html>
""", score=final_score, xai=xai_feedback)

def grade_scenario_with_gemini(question_text, transcript):
    """
    Returns (score_0_100, short_explanation)
    If analyzer.llm is missing, uses simple heuristic.
    """
    if getattr(analyzer, "llm", None):
        try:
            prompt = f"""
You are a hiring evaluator. Grade the candidate's spoken answer to the scenario.

Question: {question_text}
Answer transcript: {transcript or "(audio provided, transcript unavailable)"}

Return a JSON with keys: score (0-100 integer) and reason (short).
"""
            resp = analyzer.llm.generate_content(prompt)
            raw = resp.text.strip().strip("`")
            start = raw.find("{"); end = raw.rfind("}")
            obj = json.loads(raw[start:end+1]) if start!=-1 and end!=-1 else {}
            sc = int(obj.get("score", 70))
            rs = obj.get("reason", "Reason unavailable.")
            return sc, rs
        except Exception as e:
            print("⚠️ Gemini scenario grade failed:", e)

    # Fallback heuristic
    if transcript and len(transcript.split()) > 10:
        return 70, "Coherent response; more specifics recommended."
    return 55, "Brief/unclear answer; add steps, risks, and stakeholders."

# ------------------------------------------------------------------------------
# Results view for HR with threshold select and email to selected
# ------------------------------------------------------------------------------
@app.route("/test_results/<int:job_id>", methods=["GET"])
def test_results(job_id):
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    db = get_db()
    rows = db.execute("""
        SELECT ta.*, a.applicant_name, a.applicant_email
        FROM test_attempts ta
        JOIN test_invites ti ON ti.test_id = ta.test_id AND ti.applicant_id = ta.applicant_id
        JOIN applications a ON a.id = ta.applicant_id
        JOIN tests t ON t.id = ta.test_id
        WHERE t.job_id = ?
        ORDER BY ta.total_score DESC, ta.submitted_at DESC
    """, (job_id,)).fetchall()
    db.close()
    return render_template_string("""
<!doctype html><html><head>
<title>Test Results</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
<script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
</head><body class="p-4">
<h3>Test Results</h3>
<table class="table table-striped">
  <thead><tr><th>Candidate</th><th>Email</th><th>Score</th></tr></thead>
  <tbody>
  {% for r in rows %}
    <tr><td>{{ r['applicant_name'] }}</td><td>{{ r['applicant_email'] }}</td><td>{{ r['total_score'] }}%</td></tr>
  {% else %}
    <tr><td colspan="3" class="text-center text-muted">No results yet.</td></tr>
  {% endfor %}
  </tbody>
</table>
<hr>
<div class="form-inline">
  <label class="mr-2">Select Threshold (%)</label>
  <input id="th" type="number" class="form-control mr-2" min="0" max="100" value="70">
  <button id="btn" class="btn btn-success">Notify Selected</button>
</div>
<div id="msg" class="mt-3"></div>
<script>
$('#btn').on('click', async function(){
  const th = Number($('#th').val()||0);
  const res = await fetch('{{ url_for("notify_selected_after_test", job_id=job_id) }}', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ threshold: th })
  });
  const out = await res.json();
  $('#msg').text(out.message||JSON.stringify(out));
});
</script>
</body></html>
""", rows=rows, job_id=job_id)

@app.route("/notify_selected_after_test/<int:job_id>", methods=["POST"])
def notify_selected_after_test(job_id):
    if "user_id" not in session:
        return jsonify({"message":"Unauthorized"}), 401
    th = int((request.get_json() or {}).get("threshold", 0))
    db = get_db()
    rows = db.execute("""
        SELECT ta.*, a.applicant_name, a.applicant_email, a.id AS app_id
        FROM test_attempts ta
        JOIN test_invites ti ON ti.test_id = ta.test_id AND ti.applicant_id = ta.applicant_id
        JOIN applications a ON a.id = ta.applicant_id
        JOIN tests t ON t.id = ta.test_id
        WHERE t.job_id = ? AND ta.total_score >= ?
    """,(job_id, th)).fetchall()
    if not rows:
        db.close(); return jsonify({"message": f"No candidates above {th}%"}), 200
    sent=0
    for r in rows:
        if send_email(r["applicant_email"], "You have been selected!", 
                      f"Dear {r['applicant_name']},\n\nCongrats! You have been selected.\n\nRegards,\nTalent Team"):
            db.execute("UPDATE applications SET status='Selected' WHERE id=?", (r["app_id"],))
            sent+=1
    db.commit(); db.close()
    return jsonify({"message": f"✅ Selection emails sent to {sent} candidates ≥ {th}%."})

# ------------------------------------------------------------------------------
# Static file serving (protected)
# ------------------------------------------------------------------------------
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    # During tests, we allow candidate access without login (token protects)
    if "user_id" not in session and not request.args.get("public"):
        # For admin assets only; test pages pass ?public=1 when needed
        return "Unauthorized", 401
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)



# ───────────────────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────────────────

def get_db_connection():
    return sqlite3.connect('hr_users.db')

def send_email_smtp(to_email: str, subject: str, html_body: str) -> None:
    """
    Sends an HTML email using SMTP credentials from .env:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, FROM_EMAIL
    """
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    from_email = os.getenv("FROM_EMAIL", smtp_user)

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    msg.attach(MIMEText(html_body, 'html'))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        if smtp_user and smtp_pass:
            server.login(smtp_user, smtp_pass)
        server.sendmail(from_email, [to_email], msg.as_string())



# ───────────────────────────────────────────────────────────────────────────────
# Generate test & send link to SHORTLISTED only
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/generate_link", methods=["POST"])
def generate_link():
    if "user_id" not in session:
        return jsonify({"message": "Unauthorized"}), 401

    data = request.get_json() or {}
    job_id = data.get("job_id")
    logical = int(data.get("logical", 0))
    cognitive = int(data.get("cognitive", 0))
    scenario = int(data.get("scenario", 0))
    technical = int(data.get("technical", 0))
    topics = (data.get("topics") or "").strip()
    duration = int(data.get("duration", 30))

    if not job_id:
        return jsonify({"message": "Missing job_id"}), 400

    db = get_db()
    cur = db.cursor()

    # ✅ Create test record
    cur.execute("""
        INSERT INTO tests (job_id, total_duration)
        VALUES (?, ?)
    """, (job_id, duration))
    test_id = cur.lastrowid

    # Optional: store test blueprint (for reference)
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS test_blueprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER,
                logical INTEGER,
                cognitive INTEGER,
                scenario INTEGER,
                technical INTEGER,
                topics TEXT
            )
        """)
        cur.execute("""
            INSERT INTO test_blueprints (test_id, logical, cognitive, scenario, technical, topics)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (test_id, logical, cognitive, scenario, technical, topics))
    except Exception as e:
        print("⚠️ Blueprint insert skipped:", e)

    # ✅ Fetch shortlisted candidates — same as your notify_selected_after_test
    rows = db.execute("""
        SELECT id AS app_id, applicant_name, applicant_email
        FROM applications
        WHERE job_id = ? AND status = 'Shortlisted'
          AND applicant_email IS NOT NULL AND TRIM(applicant_email) <> ''
    """, (job_id,)).fetchall()

    if not rows:
        db.close()
        return jsonify({"message": "No shortlisted candidates found."}), 200

    # ✅ Ensure test_invites table exists (same schema)
    db.execute("""
        CREATE TABLE IF NOT EXISTS test_invites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id INTEGER,
            applicant_id INTEGER,
            token TEXT UNIQUE,
            started INTEGER DEFAULT 0,
            completed INTEGER DEFAULT 0
        )
    """)

    sent = 0
    base_url = request.url_root.rstrip("/")

    # ✅ Send test link email to each shortlisted candidate
    for r in rows:
        app_id = r["app_id"]
        email = r["applicant_email"]
        name = r["applicant_name"]
        token = uuid.uuid4().hex

        # Insert invite record
        db.execute("""
            INSERT INTO test_invites (test_id, applicant_id, token)
            VALUES (?, ?, ?)
        """, (test_id, app_id, token))

        test_link = f"{base_url}/test/{test_id}?token={token}"
        subject = "Your Technical Assessment"
        message = f"""
        Dear {name},

        You have been shortlisted for the next round. Please take your assessment using the link below:

        {test_link}

        Duration: {duration} minutes

        Regards,
        Talent Team
        """

        # ✅ Use same send_email() logic you used in notify_selected_after_test
        try:
            if send_email(email, subject, message):
                sent += 1
        except Exception as e:
            print(f"⚠️ Email failed for {email}: {e}")

    db.commit()
    db.close()

    return jsonify({"message": f"✅ Test links sent to {sent} shortlisted candidates.", "test_id": test_id})



def get_db():
    return sqlite3.connect('hr_users.db')

@app.route("/test/<int:test_id>", methods=["GET"])
def start_candidate_test(test_id):
    token = request.args.get("token", "").strip()
    if not token:
        return render_template("error.html", message="❌ Invalid test link."), 403

    db = get_db()
    db.row_factory = sqlite3.Row
    invite = db.execute("""
        SELECT ti.*, a.applicant_name
        FROM test_invites ti
        JOIN applications a ON a.id = ti.applicant_id
        WHERE ti.test_id = ? AND ti.token = ?
    """, (test_id, token)).fetchone()
    db.close()

    if not invite:
        return render_template("error.html", message="❌ Invalid or expired test link."), 403

    # Optional: mark started to prevent reuse/multiple starts
    db = get_db()
    db.execute("UPDATE test_invites SET started = 1 WHERE token = ?", (token,))
    db.commit(); db.close()

    # ✅ Redirect directly into the Gemini app (unchanged UI/logic)
    # Your combined_app should have a route like /test/<int:test_id>
    # We pass the same token so it can validate or show it if needed.
    return redirect(f"/candidate/test/{test_id}?token={token}")


@app.route("/bridge/record_attempt", methods=["POST"])
def bridge_record_attempt():
    """
    Called by the Gemini app after a candidate submits the test.
    Body must include: token, test_id, applicant_id (or derive via token),
    total_score, ai_feedback (optional JSON or text).
    """
    data = request.get_json(force=True) or {}

    token = (data.get("token") or "").strip()
    test_id = int(data.get("test_id", 0))
    total_score = int(data.get("total_score", 0))
    ai_feedback = data.get("ai_feedback", "")

    if not token or not test_id:
        return jsonify({"ok": False, "error": "Missing token or test_id"}), 400

    db = get_db()
    db.row_factory = sqlite3.Row
    invite = db.execute("""
        SELECT applicant_id FROM test_invites WHERE token=? AND test_id=?
    """, (token, test_id)).fetchone()

    if not invite:
        db.close()
        return jsonify({"ok": False, "error": "Invalid token/test"}), 403

    applicant_id = invite["applicant_id"]

    # Insert into HR-side test_attempts (your selection route depends on this)
    db.execute("""
        INSERT INTO test_attempts (test_id, applicant_id, total_score, ai_feedback)
        VALUES (?, ?, ?, ?)
    """, (test_id, applicant_id, total_score, ai_feedback))
    db.commit(); db.close()

    # Mark invite as completed (optional)
    db = get_db()
    db.execute("UPDATE test_invites SET completed = 1 WHERE token=?", (token,))
    db.commit(); db.close()

    return jsonify({"ok": True})


# Mount the Gemini app without modifying its code or DB
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/candidate': candidate_app
})


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
