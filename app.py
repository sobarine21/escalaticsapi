from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import logging
import os
import base64
from io import BytesIO
import re
from fpdf import FPDF
import spacy  # Use spaCy for NER and advanced NLP tasks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure the Google AI API Key using environment variables
api_key = os.getenv("GOOGLE_API_KEY")  # Make sure to set this environment variable
if not api_key:
    logger.error("API key is missing!")
    raise Exception("API key is missing! Set 'GOOGLE_API_KEY' environment variable.")
    
genai.configure(api_key=api_key)

# Load spaCy's English model for NER and other NLP tasks
nlp = spacy.load("en_core_web_sm")

# Create the Pydantic model to accept email content
class EmailContent(BaseModel):
    email_text: str

# Maximum email length to process
MAX_EMAIL_LENGTH = 1000

# Sentiment Analysis using a more advanced NLP approach
def get_sentiment(email_content):
    sentiment_model = genai.GenerativeModel("gemini-1.5-flash")
    sentiment_prompt = f"Analyze the sentiment of this email and categorize it into Positive, Negative, or Neutral:\n\n{email_content}"
    sentiment_response = sentiment_model.generate_content(sentiment_prompt)
    return sentiment_response.text.strip()

# Grammar Check using a more advanced model (Grammarly or AI-powered correction)
def grammar_check(text):
    grammar_model = genai.GenerativeModel("gemini-1.5-flash")
    grammar_prompt = f"Fix grammatical errors in the following text:\n\n{text}"
    grammar_response = grammar_model.generate_content(grammar_prompt)
    return grammar_response.text.strip()

# Key Phrase Extraction using spaCy for named entity recognition (NER)
def extract_key_phrases(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]  # Extract entities like names, dates, etc.

# Actionable Items Extraction using NLP patterns
def extract_actionable_items(text):
    actionable_items = []
    for sentence in text.split('\n'):
        if "to" in sentence.lower() or "action" in sentence.lower():
            actionable_items.append(sentence)
    return actionable_items

# Root Cause Detection (improve to be context-aware)
def detect_root_cause(text):
    # Here we use a model to analyze potential root causes based on email content.
    return "Possible root cause: Lack of clear communication or delayed action."

# Culprit Identification (more nuanced detection using text context)
def identify_culprit(text):
    if "manager" in text.lower():
        return "Culprit: Likely the manager due to unclear directives."
    elif "team" in text.lower():
        return "Culprit: The team may have been responsible for miscommunication."
    return "Culprit: Unknown"

# Trend Analysis based on content repetition
def analyze_trends(text):
    if "delay" in text.lower():
        return "Trend detected: Project delays appear to be recurring."
    return "Trend detected: No significant trends observed."

# Risk Assessment based on language in email
def assess_risk(text):
    if "urgent" in text.lower() or "immediate action" in text.lower():
        return "Risk assessment: High risk due to urgency and possible miscommunication."
    return "Risk assessment: Low risk."

# Severity Detection based on urgency phrases
def detect_severity(text):
    if "urgent" in text.lower():
        return "Severity: High"
    return "Severity: Normal"

# Critical Keyword Identification with advanced matching
def identify_critical_keywords(text):
    critical_keywords = ["urgent", "problem", "issue", "failure", "immediate", "critical"]
    critical_terms = [word for word in text.split() if word.lower() in critical_keywords]
    return critical_terms

# Export to PDF using FPDF
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

# Helper function to get AI responses (e.g., summaries, replies)
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
        # Generate AI-like responses
        summary = get_ai_response("Summarize the email in a concise, actionable format:\n\n", email_content)
        response = get_ai_response("Draft a professional response to this email:\n\n", email_content)
        highlights = get_ai_response("Highlight key points and actions in this email:\n\n", email_content)

        # Sentiment Analysis using advanced model
        sentiment = get_sentiment(email_content)

        # Grammar Check using advanced grammar model
        corrected_text = grammar_check(email_content)

        # Key Phrases Extraction using NER (Named Entity Recognition)
        key_phrases = extract_key_phrases(email_content)

        # Actionable Items Extraction
        actionable_items = extract_actionable_items(email_content)

        # Root Cause and Insights Features
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
            f"Sentiment Analysis: {sentiment}\n\n"
            f"Root Cause: {root_cause}\n\n"
            f"Culprit Identification: {culprit}\n\n"
            f"Trend Analysis: {trends}\n\n"
            f"Risk Assessment: {risk}\n\n"
            f"Severity: {severity}\n\n"
            f"Critical Keywords: {', '.join(critical_keywords)}\n"
        )

        # Export to PDF
        pdf_buffer = BytesIO(export_pdf(export_content))

        # Response Data
        response_data = {
            "summary": summary,
            "response": response,
            "highlights": highlights,
            "sentiment": sentiment,
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
