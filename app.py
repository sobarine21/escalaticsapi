from flask import Flask, request, jsonify
import google.generativeai as genai
import logging
import os
import re
from io import BytesIO
from fpdf import FPDF
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure the Google AI API Key securely
api_key = os.getenv("GOOGLE_API_KEY")  # Make sure to set this environment variable
if not api_key:
    logger.error("API key is missing!")
    raise Exception("API key is missing! Set 'GOOGLE_API_KEY' environment variable.")
    
genai.configure(api_key=api_key)

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

# Export to PDF
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

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

# Helper function to generate AI-like responses
def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error: {e}")
        return ""

# API endpoint to analyze email content
@app.route('/analyze-email', methods=['POST'])
def analyze_email():
    data = request.json
    email_content = data.get('email_text', "")
    
    # Initialize feature flags (for simplicity, all features are enabled here)
    features = {
        "sentiment": data.get('sentiment', True),
        "highlights": data.get('highlights', False),
        "response": data.get('response', False),
        "grammar_check": data.get('grammar_check', False),
        "key_phrases": data.get('key_phrases', False),
        "actionable_items": data.get('actionable_items', False),
        "root_cause": data.get('root_cause', False),
        "culprit_identification": data.get('culprit_identification', False),
        "trend_analysis": data.get('trend_analysis', False),
        "risk_assessment": data.get('risk_assessment', False),
        "severity_detection": data.get('severity_detection', False),
        "critical_keywords": data.get('critical_keywords', False),
        "export": data.get('export', False)
    }

    try:
        # Generate AI-like responses (using google.generativeai for content generation)
        summary = get_ai_response("Summarize the email in a concise, actionable format:\n\n", email_content)
        response = get_ai_response("Draft a professional response to this email:\n\n", email_content) if features["response"] else ""
        highlights = get_ai_response("Highlight key points and actions in this email:\n\n", email_content) if features["highlights"] else ""

        # Sentiment Analysis
        sentiment = get_sentiment(email_content)
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Prepare the response data
        response_data = {
            "summary": summary,
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment
            }
        }

        # Include other features as needed
        if features["grammar_check"]:
            corrected_text = grammar_check(email_content)
            response_data["corrected_text"] = corrected_text

        if features["key_phrases"]:
            key_phrases = extract_key_phrases(email_content)
            response_data["key_phrases"] = key_phrases

        if features["actionable_items"]:
            actionable_items = extract_actionable_items(email_content)
            response_data["actionable_items"] = actionable_items

        if features["root_cause"]:
            root_cause = detect_root_cause(email_content)
            response_data["root_cause"] = root_cause

        if features["culprit_identification"]:
            culprit = identify_culprit(email_content)
            response_data["culprit_identification"] = culprit

        if features["trend_analysis"]:
            trends = analyze_trends(email_content)
            response_data["trend_analysis"] = trends

        if features["risk_assessment"]:
            risk = assess_risk(email_content)
            response_data["risk_assessment"] = risk

        if features["severity_detection"]:
            severity = detect_severity(email_content)
            response_data["severity_detection"] = severity

        if features["critical_keywords"]:
            critical_terms = identify_critical_keywords(email_content)
            response_data["critical_keywords"] = critical_terms

        # Export options (e.g., PDF)
        if features["export"]:
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
                f"Critical Keywords: {', '.join(critical_terms)}\n"
            )
            pdf_buffer = BytesIO(export_pdf(export_content))
            response_data["pdf_export"] = pdf_buffer.getvalue().decode('latin1')

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error during email analysis: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
