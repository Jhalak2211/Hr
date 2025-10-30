# import sqlite3

# conn = sqlite3.connect('hr_users.db')
# cursor = conn.cursor()

# # Drop existing tables for a clean setup
# cursor.execute("DROP TABLE IF EXISTS applications")
# cursor.execute("DROP TABLE IF EXISTS jobs")
# cursor.execute("DROP TABLE IF EXISTS users")

# # -------------------------------
# # USERS TABLE
# # -------------------------------
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS users (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     username TEXT NOT NULL UNIQUE,
#     password TEXT NOT NULL
# )
# ''')

# # -------------------------------
# # JOBS TABLE
# # -------------------------------
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS jobs (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     job_title TEXT NOT NULL,
#     job_description TEXT NOT NULL,
#     location TEXT,
#     required_skills TEXT,
#     resume_keywords TEXT,
#     unique_link_id TEXT NOT NULL UNIQUE,
#     created_by_user_id INTEGER,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     FOREIGN KEY (created_by_user_id) REFERENCES users (id)
# )
# ''')

# # -------------------------------
# # APPLICATIONS TABLE
# # -------------------------------
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS applications (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     job_id INTEGER,
#     applicant_name TEXT NOT NULL,
#     applicant_email TEXT NOT NULL,
#     applicant_contact TEXT,
#     resume_filename TEXT NOT NULL,
#     photo_filename TEXT,
#     live_photo_filename TEXT,
#     match_score INTEGER,
#     matched_skills TEXT,
#     missing_skills TEXT,
#     ai_feedback TEXT,                -- ✅ Added column for AI explanation
#     status TEXT NOT NULL DEFAULT 'New',
#     applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     FOREIGN KEY (job_id) REFERENCES jobs (id)
# )
# ''')

# conn.commit()
# conn.close()
# print("✅ Database initialized with ai_feedback, live_photo, and status fields.")
# upgrade_tests_schema.py
import sqlite3

db = sqlite3.connect('hr_users.db')
c = db.cursor()

# Create tests table
c.execute("""
CREATE TABLE IF NOT EXISTS tests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER,
  total_duration INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Create test_questions table
c.execute("""
CREATE TABLE IF NOT EXISTS test_questions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  test_id INTEGER,
  question_type TEXT,
  question_text TEXT,
  option_a TEXT,
  option_b TEXT,
  option_c TEXT,
  option_d TEXT,
  correct_answer TEXT,
  explanation TEXT
)
""")

# Create test_invites table
c.execute("""
CREATE TABLE IF NOT EXISTS test_invites (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  test_id INTEGER,
  applicant_id INTEGER,
  token TEXT UNIQUE,
  started INTEGER DEFAULT 0,
  completed INTEGER DEFAULT 0
)
""")

# Create test_answers table
c.execute("""
CREATE TABLE IF NOT EXISTS test_answers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  invite_id INTEGER,
  question_id INTEGER,
  answer_text TEXT,
  is_audio INTEGER DEFAULT 0,
  audio_filename TEXT
)
""")

# Create test_attempts table
c.execute("""
CREATE TABLE IF NOT EXISTS test_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  test_id INTEGER,
  applicant_id INTEGER,
  total_score INTEGER,
  ai_feedback TEXT,
  submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

db.commit()
db.close()
print("✅ Test schema ready.")
