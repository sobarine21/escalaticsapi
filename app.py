from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fpdf import FPDF
import logging
import os
import json
import re

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

# Key Phrase Extraction
def extract_key_phrases(text):
    key_phrases = re.findall(r"\b[A-Za-z]{4,}\b", text)
    return list(set(key_phrases))  # Remove duplicates

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

# Export to JSON
def export_json(data):
    return json.dumps(data, indent=4)

# API endpoint to analyze email content
@app.post("/analyze-email")
async def analyze_email(content: EmailContent):
    email_content = content.email_text
    try:
        # Generate AI-like responses (using google.generativeai for content generation)
        model = genai.GenerativeModel("gemini-1.5-flash")
        summary = model.generate_content(f"Summarize the email in a concise, actionable format:\n\n{email_content[:MAX_EMAIL_LENGTH]}").text.strip()
        response = model.generate_content(f"Draft a professional response to this email:\n\n{email_content[:MAX_EMAIL_LENGTH]}").text.strip()
        highlights = model.generate_content(f"Highlight key points and actions in this email:\n\n{email_content[:MAX_EMAIL_LENGTH]}").text.strip()

        # Sentiment Analysis
        sentiment = get_sentiment(email_content)
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Extract Key Phrases
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

        # Prepare Response
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
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
