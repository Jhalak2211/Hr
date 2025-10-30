# Intelligent HR Recruitment Portal

This is a full-stack web application designed to streamline the recruitment process for HR personnel. The application allows HR to create unique job application links, and it uses AI to automatically parse, score, and provide feedback on candidate resumes submitted through these links.

## ✨ Key Features

* **Secure HR Authentication**: Separate registration and login portals for HR personnel using hashed passwords and session management.
* **Dynamic Job Posting**: HR can create new job postings with detailed descriptions, skills, and scoring keywords. The system generates a unique, shareable link for each posting.
* **Public Applicant Portal**: Candidates can access a clean, professional application form via the unique link to submit their name, contact info, resume, and photo.
* **AI-Powered Resume Analysis**:
    * **Automatic Parsing**: Extracts text content from uploaded PDF and DOCX resumes.
    * **Match Scoring**: Calculates a percentage score based on the semantic similarity between the resume's keywords and the job description's requirements.
    * **Keyword Analysis**: Identifies which required skills were found in the resume and which were missing.
    * **Generative AI Feedback**: Uses Google's Gemini model to provide a qualitative summary of the candidate's fit, estimate their experience, and offer actionable suggestions.
* **Comprehensive HR Dashboard**:
    * View all active job postings.
    * Remove job postings (which also deletes all associated applications and files).
    * Access a list of applicants for each job, sorted by application date.
* **Applicant Management**:
    * View detailed AI analysis for each candidate on a dedicated page.
    * **Shortlist** or **Discard** candidates to manage the hiring pipeline.
    * View all shortlisted candidates across all jobs on a separate page.



## 🛠️ Technology Stack

* **Backend**: Python, Flask
* **Frontend**: HTML, CSS, JavaScript (with Bootstrap for styling)
* **Database**: SQLite
* **AI & Machine Learning**:
    * **Generative AI**: `google-generativeai` (for Gemini 1.5 Flash)
    * **NLP & Parsing**: `spaCy`, `PyMuPDF`, `python-docx`
    * **Keyword Extraction**: `keybert`
    * **Semantic Similarity**: `sentence-transformers`
* **Deployment**: The app is configured for deployment using `gunicorn`.

## 🚀 Setup and Installation

Follow these steps to run the project on your local machine.

### 1. Prerequisites
* Python 3.8+
* Git

### 2. Clone the Repository
```bash
git clone [https://github.com/sanjanasandanshiv/resume-parsing.git](https://github.com/sanjanasandanshiv/resume-parsing.git)
cd resume-parsing