from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import PyPDF2
from docx import Document
from dotenv import load_dotenv
load_dotenv()
import os
import openai

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# TEXT EXTRACTION HANDLERS
# -------------------------------

def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_docx_text(file_bytes):
    file_stream = BytesIO(file_bytes)
    doc = Document(file_stream)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file_bytes, filename):
    if filename.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    elif filename.endswith(".docx"):
        return extract_docx_text(file_bytes)
    else:
        # assume .txt or unknown → try plain decode
        return file_bytes.decode("utf-8", errors="ignore")

# -------------------------------
# MAIN API ENDPOINT
# -------------------------------
@app.post("/analyze")
async def analyze_resume(jd: UploadFile = File(...), resume: UploadFile = File(...)):

    jd_bytes = await jd.read()
    resume_bytes = await resume.read()

    jd_text = extract_text(jd_bytes, jd.filename)
    resume_text = extract_text(resume_bytes, resume.filename)

    prompt = f"""
    Compare the following Job Description and Resume.

    JOB DESCRIPTION:
    {jd_text}

    RESUME:
    {resume_text}

    Output JSON with:
    - score (0-100)
    - summary (3 sentences)
    - recommended_action: Interview / Shortlist / Reject
    """

    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message

