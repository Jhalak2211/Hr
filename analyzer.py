# analyzer.py
import spacy, fitz, docx, io, re, numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()

# ========== CONFIG ==========
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
# Removed Google AI setup for now

# ========== RESUME PARSER ==========
class ResumeParser:
    def __init__(self, kw_model=None):
        # Load a lighter spaCy pipeline (disable heavy components to speed startup).
        try:
            # Disable components we don't need for simple tokenization/keyword work
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
            print("spaCy model 'en_core_web_sm' loaded (reduced components) successfully.")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Allow injection of a pre-created KeyBERT instance to avoid loading the same
        # transformer model multiple times (this is a common cause of long startup).
        if kw_model is not None:
            self.kw_model = kw_model
            print("KeyBERT instance injected into ResumeParser.")
        else:
            try:
                self.kw_model = KeyBERT()
                print("KeyBERT model loaded successfully (fallback).")
            except Exception as kw_err:
                print(f"!!!!!!!!!!!!!! FAILED TO LOAD KeyBERT MODEL !!!!!!!!!!!!!! Error: {kw_err}")
                self.kw_model = None

    def extract_text(self, file_path, file_content):
        text = ""
        file_type = ""
        print("\n--- Starting Text Extraction ---")
        print(f"File path received: {file_path}")
        try:
            if file_path.lower().endswith(".pdf"):
                file_type = "PDF"
                print(f"Attempting to extract text from PDF...")
                # Try opening the PDF
                doc = fitz.open(stream=file_content, filetype="pdf")
                print(f"PDF opened successfully. Number of pages: {len(doc)}")
                page_texts = []
                for i, page in enumerate(doc):
                    page_text = page.get_text("text", sort=True) # Added flags for better extraction
                    if page_text and page_text.strip():
                        print(f"   Extracted text from page {i+1} (snippet): '{page_text[:50].strip()}...'")
                        page_texts.append(page_text)
                    else:
                         print(f"   Page {i+1} seems to have no extractable text.")
                text = "\n".join(page_texts) # Use newline separator
                doc.close()
                if not text.strip():
                     print(">>> WARNING: PDF text extraction resulted in EMPTY string after joining pages.")
                else:
                     print(f">>> SUCCESS: Extracted {len(text)} characters from PDF.")

            elif file_path.lower().endswith(".docx"):
                file_type = "DOCX"
                print(f"Attempting to extract text from DOCX...")
                doc = docx.Document(io.BytesIO(file_content))
                print("DOCX opened successfully.")
                para_texts = [para.text for para in doc.paragraphs if para.text and para.text.strip()]
                text = "\n".join(para_texts) # Use newline separator
                if not text.strip():
                     print(">>> WARNING: DOCX text extraction resulted in EMPTY string after joining paragraphs.")
                else:
                     print(f">>> SUCCESS: Extracted {len(text)} characters from DOCX.")
            else:
                 print(f"Unsupported file type: {file_path}")
                 print("--- Finished Text Extraction (Unsupported) ---")
                 return ""

        except Exception as e:
            print(f"!!!!!!!!!!!!!! ERROR DURING TEXT EXTRACTION ({file_type}) !!!!!!!!!!!!!!")
            print(f"Error details: {e}")
            print("--- Finished Text Extraction (Error) ---")
            return "" # Return empty string on error

        print("--- Finished Text Extraction ---")
        return text

    def extract_keywords(self, text, top_n=50):
        print("\n--- Starting Keyword Extraction ---")
        if not self.kw_model:
             print("Keyword extraction failed: KeyBERT model not loaded.")
             return []
        if not text or not text.strip():
             print("Keyword extraction skipped: Input text is empty.")
             print("--- Finished Keyword Extraction (Skipped) ---")
             return []
        print(f"Extracting keywords from text (length {len(text)} chars)...")
        try:
            keywords = self.kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2), stop_words="english",
                use_mmr=True, diversity=0.7, top_n=top_n
            )
            extracted_kws = [kw for kw, score in keywords]
            print(f">>> SUCCESS: Keywords extracted: {extracted_kws}")
            print("--- Finished Keyword Extraction ---")
            return extracted_kws
        except Exception as e:
            print(f"!!!!!!!!!!!!!! ERROR DURING KEYWORD EXTRACTION !!!!!!!!!!!!!!")
            print(f"Error details: {e}")
            print("--- Finished Keyword Extraction (Error) ---")
            return []


# ======== Load the sentence transformer first, then reuse it for KeyBERT ========
try:
    # Load the sentence-transformers model once. This can still take time on first run,
    # but ensures other components reuse the in-memory model instead of re-loading it.
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("SentenceTransformer model loaded successfully.")
    # Small warm-up to reduce first-prediction latency (initial backend compilation/cache).
    try:
        _ = embedder.encode(["warmup"], show_progress_bar=False)
        print("Embedder warmup completed.")
    except Exception:
        # Non-fatal: warmup may fail in some environments; ignore.
        pass
except Exception as emb_err:
    print(f"!!!!!!!!!!!!!! FAILED TO LOAD SentenceTransformer MODEL !!!!!!!!!!!!!! Error: {emb_err}")
    embedder = None

# Create a KeyBERT instance that reuses the same sentence-transformer model to avoid
# loading the transformer twice (very common cause of slow startup and high memory).
try:
    if embedder is not None:
        kw_model = KeyBERT(model=embedder)
    else:
        kw_model = KeyBERT()
    print("KeyBERT instance created/reused successfully.")
except Exception as kw_err:
    print(f"!!!!!!!!!!!!!! FAILED TO CREATE KeyBERT INSTANCE !!!!!!!!!!!!!! Error: {kw_err}")
    kw_model = None

# Instantiate parser with injected KeyBERT instance
parser = ResumeParser(kw_model=kw_model)

# ========== ANALYSIS HELPERS ==========
def calculate_match(resume_keywords, jd_text):
    print("\n--- Starting Match Calculation ---")
    if not embedder:
        print("Match Calculation failed: SentenceTransformer model not loaded.")
        return {"score": 0, "matches": [], "misses": []}
    print(f"Resume Keywords received: {resume_keywords}")
    print(f"JD Text received (snippet): '{jd_text[:100]}...'")

    if not resume_keywords or not jd_text:
        print("Match Calculation Warning: Missing resume keywords or JD text.")
        print("--- Finished Match Calculation (Input Missing) ---")
        return {"score": 0, "matches": [], "misses": []}

    jd_keywords = parser.extract_keywords(jd_text, top_n=30)
    print(f"Extracted JD Keywords: {jd_keywords}")

    if not jd_keywords:
        print("Match Calculation Warning: Could not extract keywords from JD.")
        print("--- Finished Match Calculation (No JD Keywords) ---")
        return {"score": 0, "matches": [], "misses": []}
    if not resume_keywords: # Double check after JD extraction attempt
        print("Match Calculation Warning: Resume keywords are empty.")
        print("--- Finished Match Calculation (No Resume Keywords) ---")
        return {"score": 0, "matches": [], "misses": []}

    try:
        resume_embeddings = embedder.encode(resume_keywords)
        jd_embeddings = embedder.encode(jd_keywords)

        similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
        match_percentage, matched_skills, missing_skills = 0, [], []

        if similarity_matrix.size > 0:
            max_similarity_scores = np.max(similarity_matrix, axis=0)
            match_percentage = round(np.mean(max_similarity_scores) * 100)
            print(f"Calculated Match Percentage: {match_percentage}%")
            for i, jd_word in enumerate(jd_keywords):
                if max_similarity_scores[i] >= 0.5:
                    matched_skills.append(jd_word)
                else:
                    missing_skills.append(jd_word)
            print(f"Matched Skills: {matched_skills}")
            print(f"Missing Skills: {missing_skills}")
        else:
            print("Match Calculation Warning: Similarity matrix was empty.")

    except Exception as e:
        print(f"!!!!!!!!!!!!!! ERROR DURING MATCH CALCULATION !!!!!!!!!!!!!!")
        print(f"Error details: {e}")
        print("--- Finished Match Calculation (Error) ---")
        return {"score": 0, "matches": [], "misses": []} # Return 0 on error

    print("--- Finished Match Calculation ---")
    return {"score": match_percentage, "matches": matched_skills, "misses": missing_skills}

# Removed get_ats_feedback function