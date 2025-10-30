
# import os
# import re
# import json
# import uuid
# import random
# import sqlite3
# import requests
# from dotenv import load_dotenv
# from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
# from flask_mail import Mail, Message
# import google.generativeai as genai

# # -----------------------
# # Config + init
# # -----------------------
# load_dotenv()

# app = Flask(__name__)
# app.secret_key = os.getenv("FLASK_SECRET", "d9e5ca40541dc718b6fa283652a58884e7db086a14a6f32e1664d313145e60a7")

# # Gemini config
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# # LLM_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

# # Mail config
# app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
# app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", 587))
# app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS", "True").lower() in ("true", "1", "yes")
# app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
# app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
# app.config["MAIL_DEFAULT_SENDER"] = app.config["MAIL_USERNAME"]
# mail = Mail(app)

# # DB
# DB_FILE = "results.db"

# def init_db():
#     with sqlite3.connect(DB_FILE) as conn:
#         c = conn.cursor()
#         c.execute("""
#             CREATE TABLE IF NOT EXISTS results (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 email TEXT,
#                 name TEXT,
#                 test_id TEXT,
#                 cognitive INTEGER,
#                 logical INTEGER,
#                 technical_score INTEGER,
#                 scenario_score INTEGER,
#                 total INTEGER,
#                 scenario_explanation TEXT,
#                 technical_explanation TEXT
#             )
#         """)
#         conn.commit()

# init_db()

# # -----------------------
# # Utilities
# # -----------------------
# def normalize(text):
#     if text is None: return ""
#     return re.sub(r'\s+', ' ', str(text).strip()).lower()

# def option_index_from_label(lbl):
#     if lbl is None: return None
#     s = str(lbl).strip().lower()
#     m = re.search(r'\b([1-4])\b', s)
#     if m: return int(m.group(1)) - 1
#     m = re.search(r'\b([a-d])\b', s)
#     if m: return ord(m.group(1)) - ord('a')
#     return None

# def is_choice_correct(selected_text, correct_spec, options):
#     if selected_text is None: return False
#     sel = normalize(selected_text)
#     opts = [normalize(o) for o in options]
#     corr = normalize(correct_spec)
#     if corr in opts: return sel == corr
#     idx = option_index_from_label(corr)
#     if idx is not None and 0 <= idx < len(opts): return sel == opts[idx]
#     m = re.match(r'^([a-d])[\.\)\-:]\s*(.*)$', corr)
#     if m:
#         letter, rest = m.group(1), m.group(2)
#         idx = ord(letter) - ord('a')
#         if 0 <= idx < len(opts): return sel == opts[idx]
#     for o in opts:
#         if sel == o:
#             if corr in o or o in corr: return True
#             idx = option_index_from_label(corr)
#             if idx is not None:
#                 if opts.index(o) == idx: return True
#             return False
#     return False

# # -----------------------
# # Gemini Question Generators
# # -----------------------
# import re, json, random
# def _call_gemini_generator(prompt, num_questions, default_category, retries=2):
#     for attempt in range(retries):
#         try:
#             model = genai.GenerativeModel("gemini-2.5-flash")
#             resp = model.generate_content(prompt)

#             raw = (resp.text or "").strip()
#             print(f"\n🔍 [Gemini Attempt {attempt+1}] RAW OUTPUT:\n{raw}\n")

#             # Extract JSON safely
#             clean = re.sub(r'(?:json)?', '', raw).strip()
#             match = re.search(r'\[.*\]', clean, re.DOTALL)
#             clean_json = match.group(0) if match else clean

#             # Try parsing
#             questions = json.loads(clean_json)

#             validated = []
#             for q in questions:
#                 if (
#                     isinstance(q, dict)
#                     and q.get("question")
#                     and isinstance(q.get("options"), list)
#                     and len(q["options"]) == 4
#                     and q.get("answer")
#                 ):
#                     validated.append({
#                         "question": q["question"],
#                         "options": q["options"],
#                         "answer": q["answer"],
#                         "category": q.get("category", default_category).lower()
#                     })

#             if validated:
#                 print(f"✅ Gemini generated {len(validated)} valid questions.")
#                 return validated[:num_questions]

#             print(f"⚠️ Attempt {attempt+1} — no valid questions parsed, retrying...")

#         except Exception as e:
#             print(f"❌ Gemini generator error (attempt {attempt+1}):", e)

#     # Only use fallback if Gemini totally fails
#     print("🚨 All Gemini attempts failed — using fallback questions.")
#     return []


# def generate_cognitive_questions(num_questions: int, retries: int = 2):
#     random_seed = random.randint(1, 10000)
#     prompt = f"""
#     You are a question generator AI. Generate {num_questions} unique cognitive multiple-choice questions.
#     Avoid repeating questions or topics. Use varied math, memory, or reasoning concepts.
#     Return pure JSON only. Each question must be unique and original.
#     Use this randomness seed: {random_seed}

#     Format:
#     [
#       {{
#         "question": "...",
#         "options": ["...", "...", "...", "..."],
#         "answer": "...",
#         "category": "cognitive"
#       }}
#     ]
#     """
#     return _call_gemini_generator(prompt, num_questions, "cognitive", retries)


# def generate_logical_questions(num_questions: int, retries: int = 2):
#     prompt = f"""
#     Generate {num_questions} logical reasoning multiple-choice questions.
#     JSON only, no text, no markdown.
#     Format:
#     [
#       {{
#         "question": "...",
#         "options": ["...", "...", "...", "..."],
#         "answer": "...",
#         "category": "logical"
#       }}
#     ]
#     """
#     return _call_gemini_generator(prompt, num_questions, "logical", retries)

# def generate_technical_questions(num_questions: int, domain: str, retries: int = 2):
#     prompt = f"""
#     Generate {num_questions} technical multiple-choice questions about {domain}.
#     JSON only, no text, no markdown.
#     Format:
#     [
#       {{
#         "question": "...",
#         "options": ["...", "...", "...", "..."],
#         "answer": "...",
#         "category": "{domain.lower()}"
#       }}
#     ]
#     """
#     return _call_gemini_generator(prompt, num_questions, domain.lower(), retries)

# # -----------------------
# # Scenario-Based Module
# # -----------------------
# SCENARIO_QUESTION_POOL = [
#     {"question": "Describe a time you resolved a conflict with a teammate.", "category": "scenario"},
#     {"question": "Explain how you would handle a missed project deadline.", "category": "scenario"},
#     {"question": "How would you motivate a demotivated colleague?", "category": "scenario"},
#     {"question": "Share a situation where you made a quick decision under pressure.", "category": "scenario"},
#     {"question": "Describe how you handled receiving negative feedback.", "category": "scenario"}
# ]
# INTRO_QUESTION_POOL = [
#     {"question": "Tell me more about yourself.", "category": "intro"},
#     {"question": "What are your strengths and weaknesses?", "category": "intro"}
# ]




# def evaluate_with_llm(question, answer, category):
#     """
#     Evaluate open-ended (scenario) answers using Gemini LLM.
#     Returns a JSON with {score, explanation}, score is 0–10.
#     """
#     if not os.getenv("GEMINI_API_KEY"):
#         return {"score": 0, "explanation": "ERROR: Gemini API key not configured."}

#     if not answer.strip():
#         return {"score": 0, "explanation": "No answer provided."}

#     prompt = f"""
#     You are an experienced HR evaluator.
#     Evaluate the following candidate answer for the question below.

#     Question: "{question}"
#     Candidate's Answer: "{answer}"

#     Rate the response on a scale of **0 to 5**, where:
#       0 = No relevant or coherent answer,
#       2 = Very weak or irrelevant response,
#       5 = Acceptable, somewhat complete,
#       8 = Strong, clear, and well-structured answer,
#       10 = Excellent and insightful answer.

#     Consider factors such as relevance, clarity, depth, and professionalism.

#     Return only a valid JSON object in this format:
#     {{
#         "score": <integer from 0 to 10>,
#         "explanation": "<brief reason for the score>"
#     }}
#     """

#     try:
#         model = genai.GenerativeModel("gemini-2.5-flash")
#         resp = model.generate_content(prompt)
#         text = (resp.text or "").strip()

#         # Clean any markdown
#         text = text.replace("```json", "").replace("```", "")
#         evaluation = json.loads(text)
#         evaluation['score'] = max(0, min(10, int(evaluation.get('score', 0))))
#         return evaluation


#     except Exception as e:
#         print(f"[Gemini Evaluation Error] {e}")
#         return {"score": 0, "explanation": f"Error during evaluation: {e}"}
    
#     # try:
#     #     model = genai.GenerativeModel("gemini-2.5-flash")
#     #     resp = model.generate_content(
#     #         prompt,
#     #         # Use MIME type for reliable JSON response
#     #         config={"response_mime_type": "application/json"}
#     #     )

#     #     llm_output = resp.text.strip()
#     #     return json.loads(llm_output)

#     # except Exception as e:
#     #     print(f"⚠️ LLM evaluation failed: {e}")
#     #     return {"score": 0, "explanation": f"Error during Gemini evaluation: {e}"}



# # -----------------------
# # Routes
# # -----------------------
# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/generate_link", methods=["POST"])
# def generate_link():
#     emails_raw = request.form.get("emails", "").strip()
#     candidate_name = request.form.get("candidate_name", "").strip()
    
#     selected_types = request.form.getlist("assessment_type")
    
#     # Get number of questions for each type
#     logical_count = int(request.form.get("logical_questions_count", 0))
#     cognitive_count = int(request.form.get("cognitive_questions_count", 0))
#     technical_count = int(request.form.get("technical_questions_count", 0))
#     scenario_count = int(request.form.get("scenario_questions_count", 0))
#     technical_domain = request.form.get("technical_domain", "").strip()

#     if not emails_raw:
#         flash("Please enter candidate emails (comma separated).", "error")
#         return redirect(url_for("index"))

#     emails = [e.strip() for e in emails_raw.split(",") if e.strip()]
    
#     all_questions = []

#     # Always add introductory questions
#     all_questions.extend(INTRO_QUESTION_POOL)
    
#     # Add other question types based on form data
#     if "logical" in selected_types and logical_count > 0:
#         all_questions.extend(generate_logical_questions(logical_count))
#     if "cognitive" in selected_types and cognitive_count > 0:
#         all_questions.extend(generate_cognitive_questions(cognitive_count))
#     if "technical" in selected_types and technical_count > 0 and technical_domain:
#         all_questions.extend(generate_technical_questions(technical_count, technical_domain))
#     if "scenario" in selected_types and scenario_count > 0:
#         all_questions.extend(random.sample(SCENARIO_QUESTION_POOL, min(scenario_count, len(SCENARIO_QUESTION_POOL))))
        
#     random.shuffle(all_questions)

#     for email in emails:
#         test_id = str(uuid.uuid4())
        
#         file_name = f"test_{test_id}.json"
#         with open(file_name, "w", encoding="utf-8") as f:
#             json.dump({"email": email, "name": candidate_name, "questions": all_questions}, f, ensure_ascii=False, indent=2)

#         link = url_for("take_test", test_id=test_id, _external=True)
#         try:
#             msg = Message("Your Assessment Link", recipients=[email])
#             msg.body = f"Hello {candidate_name},\n\nPlease take your assessment here:\n\n{link}\n\nBest,\nHR Team"
#             mail.send(msg)
#             print(f"Sent email to {email} with link {link}")
#             flash(f"Sent test link to {email}", "success")
#         except Exception as e:
#             print(f"Error sending email to {email}: {e}")
#             print("Fallback link:", link)
#             flash(f"Could not send email to {email}. Link printed on server log.", "error")

#     return redirect(url_for("index"))

# @app.route("/test/<test_id>", methods=["GET", "POST"])
# def take_test(test_id):
#     file_name = f"test_{test_id}.json"
#     if not os.path.exists(file_name):
#         return "Invalid or expired test link.", 404

#     with open(file_name, "r", encoding="utf-8") as f:
#         payload = json.load(f)
#     questions = payload.get("questions", [])
#     email = payload.get("email", "")
#     name = payload.get("name", "")

#     if request.method == "POST":
#         user_email = request.form.get("email", "").strip() or email
#         cognitive_score = 0
#         logical_score = 0
#         technical_score = 0
#         scenario_score = 0
#         total_score = 0
#         scenario_count = 0   # ✅ define before using
#         scenario_explanations = []
        

#         for idx, q in enumerate(questions):
#             q_type = q.get("category", "").strip().lower()
            
#             if any(word in q_type for word in ["cognitive", "logical", "technical"]):
#                 selected = request.form.get(f"q{idx}")
#                 if is_choice_correct(selected, q.get("answer"), q.get("options", [])):
#                     total_score += 1
#                     if q_type == "cognitive":
#                         cognitive_score += 1
#                     elif q_type == "logical":
#                         logical_score += 1
#                     elif q_type == "technical":
#                         technical_score += 1
#             # For scenario-based (LLM-evaluated) questions
#             elif q_type == "scenario":
#                 answer = request.form.get(f"q{idx}")
#                 evaluation = evaluate_with_llm(q.get("question"), answer, q_type)
#                 score = evaluation.get("score", 0)  # ✅ Define score here
#                 # scenario_score += evaluation.get("score", 0)
#                 scenario_score += score
#                 scenario_count += 1
#                 scenario_explanations.append({
#                     "question": q.get("question"),
#                     "answer": answer,
#                     "score": score,
#                     "explanation": evaluation.get("explanation")
#                     # "explanation": evaluation.get("explanation"),
#                     # "score": evaluation.get("score", 0)
#                 })

#         # ✅ Average the scenario score
#         scenario_score = (scenario_score / scenario_count) if scenario_count > 0 else 0

#         # ✅ Compute total score as the sum of all section scores
#         total_score = cognitive_score + logical_score + technical_score + scenario_score

#         # ✅ Save result in database
#         with sqlite3.connect(DB_FILE) as conn:
#             c = conn.cursor()
#             c.execute(
#                 "INSERT INTO results (email, name, test_id, cognitive, logical, technical_score, scenario_score, total, scenario_explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
#                 (user_email, name, test_id, cognitive_score, logical_score, technical_score, scenario_score, total_score, json.dumps(scenario_explanations))
#             )
#             conn.commit()
#         os.remove(file_name)
#         return render_template("submit_success.html")

#     return render_template("test.html", questions=questions, test_id=test_id, email=email, name=name)

# @app.route("/dashboard")
# def dashboard():
#     with sqlite3.connect(DB_FILE) as conn:
#         conn.row_factory = sqlite3.Row
#         c = conn.cursor()
#         c.execute("SELECT * FROM results")
#         rows = c.fetchall()
    
#     results = []
#     for row in rows:
#         result_dict = dict(row)
#         if result_dict.get("scenario_explanation"):
#             result_dict["scenario_explanation"] = json.loads(result_dict["scenario_explanation"])
#         results.append(result_dict)

#     return render_template("dashboard.html", results=results)

# if __name__ == "__main__":
#     app.run(debug=True)



















# This is working perfeclty right with even technical questions scoring and Average of scenarios

# import os
# import re
# import json
# import uuid
# import random
# import sqlite3
# import requests
# from dotenv import load_dotenv
# from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
# from flask_mail import Mail, Message
# import google.generativeai as genai

# # -----------------------
# # Config + init
# # -----------------------
# load_dotenv()

# app = Flask(__name__)
# app.secret_key = os.getenv("FLASK_SECRET", "d9e5ca40541dc718b6fa283652a58884e7db086a14a6f32e1664d313145e60a7")

# # Gemini config
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# # LLM_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

# # Mail config
# app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
# app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", 587))
# app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS", "True").lower() in ("true", "1", "yes")
# app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
# app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
# app.config["MAIL_DEFAULT_SENDER"] = app.config["MAIL_USERNAME"]
# mail = Mail(app)

# # DB
# DB_FILE = "results.db"

# def init_db():
#     with sqlite3.connect(DB_FILE) as conn:
#         c = conn.cursor()
#         c.execute("""
#             CREATE TABLE IF NOT EXISTS results (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 email TEXT,
#                 name TEXT,
#                 test_id TEXT,
#                 cognitive INTEGER,
#                 logical INTEGER,
#                 technical_score INTEGER,
#                 scenario_score REAL,
#                 total REAL,
#                 scenario_explanation TEXT,
#                 technical_explanation TEXT
#             )
#         """)
#         conn.commit()

# init_db()

# # -----------------------
# # Utilities
# # -----------------------
# def normalize(text):
#     if text is None: return ""
#     return re.sub(r'\s+', ' ', str(text).strip()).lower()

# def option_index_from_label(lbl):
#     if lbl is None: return None
#     s = str(lbl).strip().lower()
#     m = re.search(r'\b([1-4])\b', s)
#     if m: return int(m.group(1)) - 1
#     m = re.search(r'\b([a-d])\b', s)
#     if m: return ord(m.group(1)) - ord('a')
#     return None

# def is_choice_correct(selected_text, correct_spec, options):
#     if selected_text is None: return False
#     sel = normalize(selected_text)
#     opts = [normalize(o) for o in options]
#     corr = normalize(correct_spec)
#     if corr in opts: return sel == corr
#     idx = option_index_from_label(corr)
#     if idx is not None and 0 <= idx < len(opts): return sel == opts[idx]
#     m = re.match(r'^([a-d])[\.\)\-:]\s*(.*)$', corr)
#     if m:
#         letter, rest = m.group(1), m.group(2)
#         idx = ord(letter) - ord('a')
#         if 0 <= idx < len(opts): return sel == opts[idx]
#     for o in opts:
#         if sel == o:
#             if corr in o or o in corr: return True
#             idx = option_index_from_label(corr)
#             if idx is not None:
#                 if opts.index(o) == idx: return True
#             return False
#     return False

# # -----------------------
# # Gemini Question Generators
# # -----------------------
# def _call_gemini_generator(prompt, num_questions, default_category, retries=2):
#     for attempt in range(retries):
#         try:
#             model = genai.GenerativeModel("gemini-2.5-flash")
#             resp = model.generate_content(prompt)

#             raw = (resp.text or "").strip()
#             print(f"\n🔍 [Gemini Attempt {attempt+1}] RAW OUTPUT:\n{raw}\n")

#             # Extract JSON safely
#             clean = re.sub(r'(?:json)?', '', raw).strip()
#             match = re.search(r'\[.*\]', clean, re.DOTALL)
#             clean_json = match.group(0) if match else clean

#             # Try parsing
#             questions = json.loads(clean_json)

#             validated = []
#             for q in questions:
#                 if (
#                     isinstance(q, dict)
#                     and q.get("question")
#                     and isinstance(q.get("options"), list)
#                     and len(q["options"]) == 4
#                     and q.get("answer")
#                 ):
#                     validated.append({
#                         "question": q["question"],
#                         "options": q["options"],
#                         "answer": q["answer"],
#                         "category": q.get("category", default_category).lower()
#                     })

#             if validated:
#                 print(f"✅ Gemini generated {len(validated)} valid questions.")
#                 return validated[:num_questions]

#             print(f"⚠️ Attempt {attempt+1} — no valid questions parsed, retrying...")

#         except Exception as e:
#             print(f"❌ Gemini generator error (attempt {attempt+1}):", e)

#     # Only use fallback if Gemini totally fails
#     print("🚨 All Gemini attempts failed — using fallback questions.")
#     return []


# def generate_cognitive_questions(num_questions: int, retries: int = 2):
#     random_seed = random.randint(1, 10000)
#     prompt = f"""
#     You are a question generator AI. Generate {num_questions} unique cognitive multiple-choice questions.
#     Avoid repeating questions or topics. Use varied math, memory, or reasoning concepts.
#     Return pure JSON only. Each question must be unique and original.
#     Use this randomness seed: {random_seed}

#     Format:
#     [
#       {{
#         "question": "...",
#         "options": ["...", "...", "...", "..."],
#         "answer": "...",
#         "category": "cognitive"
#       }}
#     ]
#     """
#     return _call_gemini_generator(prompt, num_questions, "cognitive", retries)


# def generate_logical_questions(num_questions: int, retries: int = 2):
#     prompt = f"""
#     Generate {num_questions} logical reasoning multiple-choice questions.
#     JSON only, no text, no markdown.
#     Format:
#     [
#       {{
#         "question": "...",
#         "options": ["...", "...", "...", "..."],
#         "answer": "...",
#         "category": "logical"
#       }}
#     ]
#     """
#     return _call_gemini_generator(prompt, num_questions, "logical", retries)

# def generate_technical_questions(num_questions: int, domain: str, retries: int = 2):
#     prompt = f"""
#     Generate {num_questions} technical multiple-choice questions about {domain}.
#     JSON only, no text, no markdown.
#     Format:
#     [
#       {{
#         "question": "...",
#         "options": ["...", "...", "...", "..."],
#         "answer": "...",
#         "category": "technical"
#       }}
#     ]
#     """
#     return _call_gemini_generator(prompt, num_questions, "technical", retries)

# # -----------------------
# # Scenario-Based Module
# # -----------------------
# SCENARIO_QUESTION_POOL = [
#     {"question": "Describe a time you resolved a conflict with a teammate.", "category": "scenario"},
#     {"question": "Explain how you would handle a missed project deadline.", "category": "scenario"},
#     {"question": "How would you motivate a demotivated colleague?", "category": "scenario"},
#     {"question": "Share a situation where you made a quick decision under pressure.", "category": "scenario"},
#     {"question": "Describe how you handled receiving negative feedback.", "category": "scenario"}
# ]
# INTRO_QUESTION_POOL = [
#     {"question": "Tell me more about yourself.", "category": "intro"},
#     {"question": "What are your strengths and weaknesses?", "category": "intro"}
# ]




# def evaluate_with_llm(question, answer, category):
#     """
#     Evaluate open-ended (scenario) answers using Gemini LLM.
#     Returns a JSON with {score, explanation}, score is 0–10.
#     """
#     if not os.getenv("GEMINI_API_KEY"):
#         return {"score": 0, "explanation": "ERROR: Gemini API key not configured."}

#     if not answer.strip():
#         return {"score": 0, "explanation": "No answer provided."}

#     prompt = f"""
#     You are an experienced HR evaluator.
#     Evaluate the following candidate answer for the question below.

#     Question: "{question}"
#     Candidate's Answer: "{answer}"

#     Rate the response on a scale of **0 to 10**, where:
#       0 = No relevant or coherent answer,
#       2 = Very weak or irrelevant response,
#       5 = Acceptable, somewhat complete,
#       8 = Strong, clear, and well-structured answer,
#       10 = Excellent and insightful answer.

#     Consider factors such as relevance, clarity, depth, and professionalism.

#     Return only a valid JSON object in this format:
#     {{
#         "score": <integer from 0 to 10>,
#         "explanation": "<brief reason for the score>"
#     }}
#     """

#     try:
#         model = genai.GenerativeModel("gemini-2.5-flash")
#         resp = model.generate_content(prompt)
#         text = (resp.text or "").strip()

#         # Clean any markdown
#         text = text.replace("```json", "").replace("```", "")
#         evaluation = json.loads(text)
#         evaluation['score'] = max(0, min(10, int(evaluation.get('score', 0))))
#         return evaluation


#     except Exception as e:
#         print(f"[Gemini Evaluation Error] {e}")
#         return {"score": 0, "explanation": f"Error during evaluation: {e}"}



# # -----------------------
# # Routes
# # -----------------------
# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/generate_link", methods=["POST"])
# def generate_link():
#     emails_raw = request.form.get("emails", "").strip()
#     candidate_name = request.form.get("candidate_name", "").strip()
    
#     selected_types = request.form.getlist("assessment_type")
    
#     # Get number of questions for each type
#     logical_count = int(request.form.get("logical_questions_count", 0))
#     cognitive_count = int(request.form.get("cognitive_questions_count", 0))
#     technical_count = int(request.form.get("technical_questions_count", 0))
#     scenario_count = int(request.form.get("scenario_questions_count", 0))
#     technical_domain = request.form.get("technical_domain", "").strip()

#     if not emails_raw:
#         flash("Please enter candidate emails (comma separated).", "error")
#         return redirect(url_for("index"))

#     emails = [e.strip() for e in emails_raw.split(",") if e.strip()]
    
#     all_questions = []

#     # Always add introductory questions
#     all_questions.extend(INTRO_QUESTION_POOL)
    
#     # Add other question types based on form data
#     if "logical" in selected_types and logical_count > 0:
#         all_questions.extend(generate_logical_questions(logical_count))
#     if "cognitive" in selected_types and cognitive_count > 0:
#         all_questions.extend(generate_cognitive_questions(cognitive_count))
#     if "technical" in selected_types and technical_count > 0 and technical_domain:
#         all_questions.extend(generate_technical_questions(technical_count, technical_domain))
#     if "scenario" in selected_types and scenario_count > 0:
#         all_questions.extend(random.sample(SCENARIO_QUESTION_POOL, min(scenario_count, len(SCENARIO_QUESTION_POOL))))
        
#     random.shuffle(all_questions)

#     for email in emails:
#         test_id = str(uuid.uuid4())
        
#         file_name = f"test_{test_id}.json"
#         with open(file_name, "w", encoding="utf-8") as f:
#             json.dump({"email": email, "name": candidate_name, "questions": all_questions}, f, ensure_ascii=False, indent=2)

#         link = url_for("take_test", test_id=test_id, _external=True)
#         try:
#             msg = Message("Your Assessment Link", recipients=[email])
#             msg.body = f"Hello {candidate_name},\n\nPlease take your assessment here:\n\n{link}\n\nBest,\nHR Team"
#             mail.send(msg)
#             print(f"Sent email to {email} with link {link}")
#             flash(f"Sent test link to {email}", "success")
#         except Exception as e:
#             print(f"Error sending email to {email}: {e}")
#             print("Fallback link:", link)
#             flash(f"Could not send email to {email}. Link printed on server log.", "error")

#     return redirect(url_for("index"))

# @app.route("/test/<test_id>", methods=["GET", "POST"])
# def take_test(test_id):
#     file_name = f"test_{test_id}.json"
#     if not os.path.exists(file_name):
#         return "Invalid or expired test link.", 404

#     with open(file_name, "r", encoding="utf-8") as f:
#         payload = json.load(f)
#     questions = payload.get("questions", [])
#     email = payload.get("email", "")
#     name = payload.get("name", "")

#     if request.method == "POST":
#         user_email = request.form.get("email", "").strip() or email
#         cognitive_score = 0
#         logical_score = 0
#         technical_score = 0
#         scenario_score_sum = 0 # Using sum for scenario scores
#         scenario_count = 0 
#         scenario_explanations = []
        

#         for idx, q in enumerate(questions):
#             q_type = q.get("category", "").strip().lower()
            
#             if any(word in q_type for word in ["cognitive", "logical", "technical"]):
#                 selected = request.form.get(f"q{idx}")
#                 if is_choice_correct(selected, q.get("answer"), q.get("options", [])):
#                     # total_score += 1 # Removed as total_score is calculated at the end
#                     if q_type == "cognitive":
#                         cognitive_score += 1
#                     elif q_type == "logical":
#                         logical_score += 1
#                     elif q_type == "technical":
#                         technical_score += 1 # Technical score is correctly incremented here
#             # For scenario-based (LLM-evaluated) questions
#             elif q_type == "scenario":
#                 answer = request.form.get(f"q{idx}")
#                 evaluation = evaluate_with_llm(q.get("question"), answer, q_type)
#                 score = evaluation.get("score", 0)
#                 scenario_score_sum += score # Accumulate sum for averaging
#                 scenario_count += 1
#                 scenario_explanations.append({
#                     "question": q.get("question"),
#                     "answer": answer,
#                     "score": score,
#                     "explanation": evaluation.get("explanation")
#                 })

#         # Calculate the final scenario score (average)
#         scenario_score = (scenario_score_sum / scenario_count) if scenario_count > 0 else 0.0

#         # Compute total score as the sum of all section scores
#         total_score = float(cognitive_score + logical_score + technical_score) + scenario_score

#         # Save result in database
#         with sqlite3.connect(DB_FILE) as conn:
#             c = conn.cursor()
#             # Note: technical_explanation column is unused in the INSERT statement
#             c.execute(
#                 "INSERT INTO results (email, name, test_id, cognitive, logical, technical_score, scenario_score, total, scenario_explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
#                 (user_email, name, test_id, cognitive_score, logical_score, technical_score, scenario_score, total_score, json.dumps(scenario_explanations))
#             )
#             conn.commit()
#         os.remove(file_name)
#         return render_template("submit_success.html")

#     return render_template("test.html", questions=questions, test_id=test_id, email=email, name=name)

# @app.route("/dashboard")
# def dashboard():
#     with sqlite3.connect(DB_FILE) as conn:
#         conn.row_factory = sqlite3.Row
#         c = conn.cursor()
#         c.execute("SELECT * FROM results")
#         rows = c.fetchall()
    
#     results = []
#     for row in rows:
#         result_dict = dict(row)
#         if result_dict.get("scenario_explanation"):
#             result_dict["scenario_explanation"] = json.loads(result_dict["scenario_explanation"])
#         results.append(result_dict)

#     return render_template("dashboard.html", results=results)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# @app.route("/upload_audio", methods=["POST"])
# def upload_audio():
#     """
#     Handles audio file uploads from candidates.
#     Saves audio file and optionally sends it to Gemini for transcription/evaluation.
#     """
#     if "audio" not in request.files:
#         return jsonify({"success": False, "message": "No audio file found in request."}), 400

#     audio_file = request.files["audio"]
#     if audio_file.filename == "":
#         return jsonify({"success": False, "message": "Empty filename."}), 400

#     # Save the file locally
#     file_id = str(uuid.uuid4())
#     filename = f"{file_id}_{audio_file.filename}"
#     save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     audio_file.save(save_path)

#     print(f"✅ Audio uploaded: {save_path}")

#     # Optional: Transcribe or analyze with Gemini
#     try:
#         model = genai.GenerativeModel("gemini-2.5-flash")
#         with open(save_path, "rb") as f:
#             resp = model.generate_content(
#                 [f, "Transcribe this interview response into text."]
#             )
#         transcription = (resp.text or "").strip()
#     except Exception as e:
#         print(f"⚠️ Transcription failed: {e}")
#         transcription = None

#     return jsonify({
#         "success": True,
#         "file": filename,
#         "transcription": transcription or "Transcription unavailable."
#     })


# if __name__ == "__main__":
#     app.run(debug=True)







import os
import re
import json
import uuid
import random
import sqlite3
import time
import importlib.util
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_mail import Mail, Message
import tensorflow as tf
from tensorflow.keras.models import load_model
import google.generativeai as genai

# -----------------------
# Config + init
# -----------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "super_secret_key")

# Mail config
app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", 587))
app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS", "True").lower() in ("true", "1", "yes")
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = app.config["MAIL_USERNAME"]
mail = Mail(app)

# Gemini config (for transcription)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# File upload config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_AUDIO_EXTS = {"webm", "wav", "mp3", "m4a", "ogg"}

# -----------------------
# Database
# -----------------------
DB_FILE = "results.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                name TEXT,
                test_id TEXT,
                cognitive INTEGER,
                logical INTEGER,
                technical_score INTEGER,
                scenario_score REAL,
                introduction_score REAL,
                total REAL,
                scenario_explanation TEXT,
                introduction_explanation TEXT,
                technical_explanation TEXT
            )
        """)
        conn.commit()

init_db()

# -----------------------
# Load ML Models
# -----------------------
CONF_MODEL_PATH = "best_confidence_model.keras"
POSITIVE_MODEL_PATH = "positive.py"

print("🔄 Loading local models...")
confidence_model = load_model(CONF_MODEL_PATH)

# Dynamically import positive.py
spec = importlib.util.spec_from_file_location("positive_module", POSITIVE_MODEL_PATH)
positive_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(positive_module)

# Wrapper to predict confidence
def predict_confidence(text: str) -> float:
    # Dummy preprocessing (replace with yours)
    seq = np.array([[len(text.split()) / 50]])  # example feature
    try:
        conf = float(confidence_model.predict(seq, verbose=0)[0][0])
    except Exception:
        conf = 0.5
    return max(0.0, min(1.0, conf))

# Wrapper to predict sentiment using positive.py
def predict_sentiment(text: str) -> str:
    try:
        if hasattr(positive_module, "classifier"):
            result = positive_module.classifier(text)
            label = result[0]["label"].lower()
            return "positive" if "pos" in label else "negative"
        elif hasattr(positive_module, "predict_sentiment"):
            return positive_module.predict_sentiment(text)
    except Exception as e:
        print(f"⚠️ Sentiment prediction error: {e}")
    return "neutral"

# -----------------------
# Question pools
# -----------------------
SCENARIO_QUESTION_POOL = [
    {"question": "Describe a time you resolved a conflict with a teammate.", "category": "scenario"},
    {"question": "Explain how you would handle a missed project deadline.", "category": "scenario"},
    {"question": "How would you motivate a demotivated colleague?", "category": "scenario"},
    {"question": "Share a situation where you made a quick decision under pressure.", "category": "scenario"},
    {"question": "Describe how you handled receiving negative feedback.", "category": "scenario"}
]

INTRO_QUESTION_POOL = [
    {"question": "Tell me more about yourself.", "category": "intro"},
    {"question": "What are your strengths and weaknesses?", "category": "intro"}
]

# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_link", methods=["POST"])
def generate_link():
    emails_raw = request.form.get("emails", "").strip()
    candidate_name = request.form.get("candidate_name", "").strip()
    selected_types = request.form.getlist("assessment_type")

    logical_count = int(request.form.get("logical_questions_count", 0))
    cognitive_count = int(request.form.get("cognitive_questions_count", 0))
    technical_count = int(request.form.get("technical_questions_count", 0))
    scenario_count = int(request.form.get("scenario_questions_count", 0))
    technical_domain = request.form.get("technical_domain", "").strip()

    if not emails_raw:
        flash("Please enter candidate emails.", "error")
        return redirect(url_for("index"))

    emails = [e.strip() for e in emails_raw.split(",") if e.strip()]
    all_questions = INTRO_QUESTION_POOL.copy()

    # Add selected sections
    if "logical" in selected_types and logical_count > 0:
        all_questions.extend(generate_dummy_questions(logical_count, "logical"))
    if "cognitive" in selected_types and cognitive_count > 0:
        all_questions.extend(generate_dummy_questions(cognitive_count, "cognitive"))
    if "technical" in selected_types and technical_count > 0:
        all_questions.extend(generate_dummy_questions(technical_count, "technical"))
    if "scenario" in selected_types and scenario_count > 0:
        all_questions.extend(random.sample(SCENARIO_QUESTION_POOL, min(scenario_count, len(SCENARIO_QUESTION_POOL))))
    random.shuffle(all_questions)

    for email in emails:
        test_id = str(uuid.uuid4())
        file_name = f"test_{test_id}.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump({"email": email, "name": candidate_name, "questions": all_questions}, f, ensure_ascii=False, indent=2)

        link = url_for("take_test", test_id=test_id, _external=True)
        try:
            msg = Message("Your Assessment Link", recipients=[email])
            msg.body = f"Hello {candidate_name},\nPlease take your test here:\n{link}\n\nBest,\nHR Team"
            mail.send(msg)
        except Exception as e:
            print("Email send error:", e)

    flash("Links generated and sent successfully!", "success")
    return redirect(url_for("index"))

def generate_dummy_questions(count, category):
    return [{"question": f"Sample {category} question {i+1}", "options": ["A", "B", "C", "D"], "answer": "A", "category": category} for i in range(count)]

@app.route("/test/<test_id>", methods=["GET", "POST"])
def take_test(test_id):
    file_name = f"test_{test_id}.json"
    if not os.path.exists(file_name):
        return "Invalid test link.", 404
    with open(file_name, "r", encoding="utf-8") as f:
        payload = json.load(f)

    questions = payload["questions"]
    email = payload["email"]
    name = payload["name"]

    if request.method == "POST":
        cognitive = logical = technical = 0
        scenario_score = intro_score = 0.0
        scenario_explanations = []
        introduction_explanations = []

        for idx, q in enumerate(questions):
            q_type = q["category"]
            if q_type in ["cognitive", "logical", "technical"]:
                selected = request.form.get(f"q{idx}")
                if selected and selected.lower() == q["answer"].lower():
                    if q_type == "cognitive": cognitive += 1
                    elif q_type == "logical": logical += 1
                    elif q_type == "technical": technical += 1
            elif q_type == "intro":
                answer = request.form.get(f"q{idx}", "")
                confidence = predict_confidence(answer)
                sentiment = predict_sentiment(answer)
                base = 5 * confidence
                sentiment_bonus = 3 if sentiment == "positive" else 1 if sentiment == "neutral" else 0
                final = round(min(10, base + sentiment_bonus), 2)
                intro_score += final
                introduction_explanations.append({
                    "question": q["question"],
                    "answer": answer,
                    "confidence": confidence,
                    "sentiment": sentiment,
                    "score": final
                })
            elif q_type == "scenario":
                answer = request.form.get(f"q{idx}", "")
                scenario_explanations.append({"question": q["question"], "answer": answer, "score": 8, "explanation": "Scenario handled."})
                scenario_score += 8

        intro_score = round(intro_score / max(1, len(introduction_explanations)), 2)
        scenario_score = round(scenario_score / max(1, len(scenario_explanations)), 2)
        total = cognitive + logical + technical + intro_score + scenario_score

        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("""INSERT INTO results 
                (email, name, test_id, cognitive, logical, technical_score, scenario_score, introduction_score, total, scenario_explanation, introduction_explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (email, name, test_id, cognitive, logical, technical, scenario_score, intro_score, total, json.dumps(scenario_explanations), json.dumps(introduction_explanations)))
            conn.commit()
        os.remove(file_name)
        return render_template("submit_success.html")

    return render_template("test.html", questions=questions, test_id=test_id, email=email, name=name)

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"success": False, "message": "No audio file."}), 400
    audio = request.files["audio"]
    if not audio.filename:
        return jsonify({"success": False, "message": "Empty filename."}), 400

    test_id = request.form.get("test_id")
    email = request.form.get("email")
    ext = audio.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        ext = "webm"

    filename = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    audio.save(path)

    print(f"✅ Saved audio: {path}")

    transcription = ""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        with open(path, "rb") as f:
            resp = model.generate_content([f, "Transcribe this audio to text."])
        transcription = (resp.text or "").strip()
    except Exception as e:
        print("⚠️ Transcription failed:", e)

    confidence = predict_confidence(transcription)
    sentiment = predict_sentiment(transcription)
    base = 5 * confidence
    sentiment_bonus = 3 if sentiment == "positive" else 1 if sentiment == "neutral" else 0
    score = round(min(10, base + sentiment_bonus), 2)

    result = {
        "transcription": transcription,
        "confidence": confidence,
        "sentiment": sentiment,
        "score": score,
        "explanation": f"Confidence={confidence:.2f}, Sentiment={sentiment}, Final={score}/10"
    }

    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("UPDATE results SET introduction_score=?, introduction_explanation=? WHERE test_id=? AND email=?",
                  (score, json.dumps(result), test_id, email))
        conn.commit()

    return jsonify({"success": True, "message": "Audio evaluated.", "evaluation": result})

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/dashboard")
def dashboard():
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.cursor().execute("SELECT * FROM results").fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("scenario_explanation"):
            d["scenario_explanation"] = json.loads(d["scenario_explanation"])
        if d.get("introduction_explanation"):
            try:
                d["introduction_explanation"] = json.loads(d["introduction_explanation"])
            except:
                pass
        results.append(d)
    return render_template("dashboard.html", results=results)





import requests, os

# Base URL of your HR app
HR_BASE_URL = os.getenv("HR_BASE_URL", "http://localhost:5000")

def _notify_hr_bridge(token: str, test_id: int, total_score: int, ai_feedback: str):
    """
    Sends the test result summary back to the HR database
    so that HR can see results and run /notify_selected_after_test.
    """
    try:
        requests.post(f"{HR_BASE_URL}/bridge/record_attempt", json={
            "token": token,
            "test_id": test_id,
            "total_score": total_score,
            "ai_feedback": ai_feedback
        }, timeout=5)
    except Exception as e:
        print("⚠️ Could not notify HR bridge:", e)




if __name__ == "__main__":
    app.run(debug=True)
