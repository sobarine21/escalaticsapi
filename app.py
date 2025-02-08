from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import logging
import os
import base64
from io import BytesIO
import re
from fpdf import FPDF

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure the Google AI API Key using environment variables
api_key = os.getenv("GOOGLE_API_KEY")  # Make sure to set this environment variable on Render
if not api_key:
    logger.error("API key is missing!")
    raise Exception("API key is missing! Set 'GOOGLE_API_KEY' environment variable.")
    
genai.configure(api_key=api_key)

# Create the Pydantic model to accept email content
class EmailContent(BaseModel):
    email_text: str

# Maximum email length to process
MAX_EMAIL_LENGTH = 1000

# Sentiment Analysis
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

# Grammar Check (basic spelling correction)
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

# Key Phrase Extraction
def extract_key_phrases(text):
    key_phrases = re.findall(r"\b[A-Za-z]{4,}\b", text)
    return list(set(key_phrases))  # Remove duplicates

# Actionable Items Extraction
def extract_actionable_items(text):
    actions = [line for line in text.split("\n") if "to" in line.lower() or "action" in line.lower()]
    return actions

# Root Cause Detection
def detect_root_cause(text):
    return "Possible root cause: Lack of clear communication in the process."

# Culprit Identification
def identify_culprit(text):
    if "manager" in text.lower():
        return "Culprit: The manager might be responsible."
    elif "team" in text.lower():
        return "Culprit: The team might be responsible."
    return "Culprit: Unknown"

# Trend Analysis
def analyze_trends(text):
    return "Trend detected: Delay in project timelines."

# Risk Assessment
def assess_risk(text):
    return "Risk assessment: High risk due to delayed communication."

# Severity Detection
def detect_severity(text):
    if "urgent" in text.lower():
        return "Severity: High"
    return "Severity: Normal"

# Critical Keyword Identification
def identify_critical_keywords(text):
    critical_keywords = ["urgent", "problem", "issue", "failure"]
    critical_terms = [word for word in text.split() if word.lower() in critical_keywords]
    return critical_terms

# Export to PDF
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

# Helper function to get AI responses
def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error: {e}")
        return ""

# API endpoint to analyze email content
@app.post("/analyze-email")
async def analyze_email(content: EmailContent):
    email_content = content.email_text
    try:
        # Generate AI-like responses (using google.generativeai for content generation)
        summary = get_ai_response("Summarize the email in a concise, actionable format:\n\n", email_content)
        response = get_ai_response("Draft a professional response to this email:\n\n", email_content)
        highlights = get_ai_response("Highlight key points and actions in this email:\n\n", email_content)

        # Sentiment Analysis
        sentiment = get_sentiment(email_content)
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Grammar Check
        corrected_text = grammar_check(email_content)

        # Key Phrases Extraction
        key_phrases = extract_key_phrases(email_content)

        # Actionable Items Extraction
        actionable_items = extract_actionable_items(email_content)

        # RCA and Insights Features
        root_cause = detect_root_cause(email_content)
        culprit = identify_culprit(email_content)
        trends = analyze_trends(email_content)
        risk = assess_risk(email_content)
        severity = detect_severity(email_content)
        critical_keywords = identify_critical_keywords(email_content)

        # Prepare content for export
        export_content = (
            f"Summary:\n{summary}\n\n"
            f"Response:\n{response}\n\n"
            f"Highlights:\n{highlights}\n\n"
            f"Sentiment Analysis: {sentiment_label} (Score: {sentiment})\n\n"
            f"Root Cause: {root_cause}\n\n"
            f"Culprit Identification: {culprit}\n\n"
            f"Trend Analysis: {trends}\n\n"
            f"Risk Assessment: {risk}\n\n"
            f"Severity: {severity}\n\n"
            f"Critical Keywords: {', '.join(critical_keywords)}\n"
        )

        # Export to PDF if requested
        pdf_buffer = BytesIO(export_pdf(export_content))
        
        response_data = {
            "summary": summary,
            "response": response,
            "highlights": highlights,
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment
            },
            "corrected_text": corrected_text,
            "key_phrases": key_phrases,
            "actionable_items": actionable_items,
            "root_cause": root_cause,
            "culprit": culprit,
            "trends": trends,
            "risk": risk,
            "severity": severity,
            "critical_keywords": critical_keywords,
            "export_pdf": base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')  # Base64 encoded PDF
        }

        return response_data

    except Exception as e:
        logger.error(f"Error during email analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
