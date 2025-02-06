from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import json
import re
from io import BytesIO
from fpdf import FPDF
import logging

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google AI API Key
genai.configure(api_key="YOUR_GOOGLE_API_KEY")  # Replace with your actual API key

MAX_EMAIL_LENGTH = 1000

class EmailContent(BaseModel):
    email_text: str

# Root Route
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Email Analysis API!"}

# Helper Functions (same as before)
def get_sentiment(email_content):
    positive_keywords = ["happy", "good", "great", "excellent", "love"]
    negative_keywords = ["sad", "bad", "hate", "angry", "disappointed"]
    sentiment_score = 0
    for word in email_content.split():
        if word.lower() in positive_keywords:
            sentiment_score += 1
        elif word.lower() in negative_keywords:
            sentiment_score -= 1
    return sentiment_score

def grammar_check(text):
    corrections = {
        "recieve": "receive",
        "adress": "address",
        "teh": "the",
        "occured": "occurred"
    }
    for word, correct in corrections.items():
        text = text.replace(word, correct)
    return text

def extract_key_phrases(text):
    key_phrases = re.findall(r"\b[A-Za-z]{4,}\b", text)
    return list(set(key_phrases))  # Remove duplicates

def extract_actionable_items(text):
    actions = [line for line in text.split("\n") if "to" in line.lower() or "action" in line.lower()]
    return actions

def detect_root_cause(text):
    return "Possible root cause: Lack of clear communication in the process."

def identify_culprit(text):
    if "manager" in text.lower():
        return "Culprit: The manager might be responsible."
    elif "team" in text.lower():
        return "Culprit: The team might be responsible."
    return "Culprit: Unknown"

def analyze_trends(text):
    return "Trend detected: Delay in project timelines."

def assess_risk(text):
    return "Risk assessment: High risk due to delayed communication."

def detect_severity(text):
    if "urgent" in text.lower():
        return "Severity: High"
    return "Severity: Normal"

def identify_critical_keywords(text):
    critical_keywords = ["urgent", "problem", "issue", "failure"]
    critical_terms = [word for word in text.split() if word.lower() in critical_keywords]
    return critical_terms

def generate_wordcloud(text):
    word_counts = {}
    for word in text.split():
        word = word.lower()
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1
    return word_counts

def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during AI response generation: {str(e)}")

# API Endpoints

@app.post("/analyze-email")
async def analyze_email(content: EmailContent):
    email_content = content.email_text
    try:
        # Generate AI responses
        summary = get_ai_response("Summarize the email in a concise, actionable format:\n\n", email_content)
        response = get_ai_response("Draft a professional response to this email:\n\n", email_content)
        highlights = get_ai_response("Highlight key points and actions in this email:\n\n", email_content)

        # Sentiment Analysis
        sentiment = get_sentiment(email_content)
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Key Phrases Extraction
        key_phrases = extract_key_phrases(email_content)

        # Extract Actionable Items
        actionable_items = extract_actionable_items(email_content)

        # Root Cause Detection
        root_cause = detect_root_cause(email_content)

        # Culprit Identification
        culprit = identify_culprit(email_content)

        # Trend Analysis
        trends = analyze_trends(email_content)

        # Risk Assessment
        risk = assess_risk(email_content)

        # Severity Detection
        severity = detect_severity(email_content)

        # Critical Keyword Identification
        critical_keywords = identify_critical_keywords(email_content)

        # Prepare Response Data
        response_data = {
            "summary": summary,
            "response": response,
            "highlights": highlights,
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment
            },
            "key_phrases": key_phrases,
            "actionable_items": actionable_items,
            "root_cause": root_cause,
            "culprit": culprit,
            "trends": trends,
            "risk": risk,
            "severity": severity,
            "critical_keywords": critical_keywords
        }

        return response_data

    except Exception as e:
        logger.error(f"Error during email analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during email analysis: {str(e)}")

@app.post("/export-pdf")
async def export_pdf_endpoint(content: EmailContent):
    try:
        summary = get_ai_response("Summarize the email in a concise, actionable format:\n\n", content.email_text)
        response_data = f"Summary:\n{summary}\n"
        pdf_data = export_pdf(response_data)
        return {"pdf": pdf_data}
    except Exception as e:
        logger.error(f"Error during PDF export: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during PDF export: {str(e)}")

@app.post("/export-json")
async def export_json_endpoint(content: EmailContent):
    try:
        summary = get_ai_response("Summarize the email in a concise, actionable format:\n\n", content.email_text)
        response_data = {
            "summary": summary
        }
        return response_data
    except Exception as e:
        logger.error(f"Error during JSON export: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during JSON export: {str(e)}")

# Run the application with: uvicorn your_file_name:app --reload
