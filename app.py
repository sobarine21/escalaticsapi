from flask import Flask, request, jsonify
import google.generativeai as genai
import json
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from fpdf import FPDF
import langdetect
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Initialize Flask app
app = Flask(__name__)

# Configure API Key for Google Generative AI
genai.configure(api_key="your_google_api_key_here")  # Replace with your actual API Key

MAX_EMAIL_LENGTH = 1000

# Helper functions

def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        return str(e)

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

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

def extract_actionable_items(text):
    actions = [line for line in text.split("\n") if "to" in line.lower() or "action" in line.lower()]
    return actions

def detect_root_cause(text):
    if "lack of communication" in text.lower():
        return "Root Cause: Lack of communication between teams."
    elif "delayed response" in text.lower():
        return "Root Cause: Delayed response from the team."
    return "Root Cause: Unknown"

def assess_risk(text):
    if "urgent" in text.lower():
        return "Risk Assessment: High risk due to urgency of the issue."
    return "Risk Assessment: Normal risk."

def detect_severity(text):
    if "urgent" in text.lower():
        return "Severity: High"
    return "Severity: Normal"

def identify_critical_keywords(text):
    critical_keywords = ["urgent", "problem", "issue", "failure"]
    critical_terms = [word for word in text.split() if word.lower() in critical_keywords]
    return critical_terms

def detect_escalation_trigger(text):
    if "escalate" in text.lower() or "critical" in text.lower():
        return "Escalation Trigger: Immediate escalation required."
    return "Escalation Trigger: No immediate escalation needed."

def identify_culprit(text):
    if "manager" in text.lower():
        return "Culprit: The manager might be responsible."
    elif "team" in text.lower():
        return "Culprit: The team might be responsible."
    return "Culprit: Unknown"

def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang
    except Exception as e:
        return "Unknown"

def entity_recognition(text):
    entities = ["Email", "Action", "Team", "Manager"]
    return entities

def response_time_analysis(text):
    if "responded" in text.lower():
        return "Response Time Analysis: Response time is within acceptable range."
    return "Response Time Analysis: Response time is not clear."

def attachment_analysis(text):
    if "attached" in text.lower():
        return "Attachment Analysis: The email contains an attachment."
    return "Attachment Analysis: No attachment found."

def customer_tone_analysis(text):
    positive_tone_keywords = ["thank you", "appreciate", "grateful"]
    negative_tone_keywords = ["disappointed", "frustrated", "unhappy"]
    tone = "Neutral"
    if any(word in text.lower() for word in positive_tone_keywords):
        tone = "Positive"
    elif any(word in text.lower() for word in negative_tone_keywords):
        tone = "Negative"
    return f"Customer Tone Analysis: {tone}"

def department_identification(text):
    if "sales" in text.lower():
        return "Department Identification: Sales"
    elif "support" in text.lower():
        return "Department Identification: Support"
    return "Department Identification: Unknown"

def identify_priority(text):
    if "high priority" in text.lower():
        return "Priority: High"
    elif "low priority" in text.lower():
        return "Priority: Low"
    return "Priority: Normal"

def assess_urgency(text):
    if "urgent" in text.lower():
        return "Urgency Assessment: High urgency."
    return "Urgency Assessment: Normal urgency."

def action_item_priority(text):
    if "urgent" in text.lower() or "immediate" in text.lower():
        return "Action Item Priority: High"
    return "Action Item Priority: Normal"

def detect_deadline(text):
    deadlines = ["due", "deadline", "by"]
    if any(word in text.lower() for word in deadlines):
        return "Deadline Detection: Contains a deadline."
    return "Deadline Detection: No deadline mentioned."

def email_chain_analysis(text):
    if "forwarded" in text.lower() or "re:" in text.lower():
        return "Email Chain Analysis: This is part of an email chain."
    return "Email Chain Analysis: This email is standalone."

def executive_summary(text):
    return f"Executive Summary: {text[:200]}..."

def actionable_resolution(text):
    if "resolve" in text.lower() or "solution" in text.lower():
        return "Actionable Resolution: The email includes a resolution or solution."
    return "Actionable Resolution: No actionable resolution found."

def response_completeness(text):
    if "thank you" in text.lower() and "best regards" in text.lower():
        return "Response Completeness: Response is complete."
    return "Response Completeness: Response is incomplete."

def agreement_identification(text):
    if "agree" in text.lower():
        return "Agreement Identification: The email includes an agreement."
    return "Agreement Identification: No agreement found."

def feedback_analysis(text):
    if "feedback" in text.lower():
        return "Feedback Analysis: Feedback present."
    return "Feedback Analysis: No feedback found."

def threat_detection(text):
    threat_keywords = ["threat", "warning", "danger"]
    if any(word in text.lower() for word in threat_keywords):
        return "Threat Detection: Threat detected."
    return "Threat Detection: No threat detected."

def response_quality_assessment(text):
    if len(text.split()) < 20:
        return "Response Quality: Incomplete response."
    return "Response Quality: Complete response."

def visualize_word_frequency(word_counts):
    plt.figure(figsize=(10, 5))
    plt.bar(word_counts.keys(), word_counts.values())
    plt.xticks(rotation=45)
    plt.title("Word Frequency")
    plt.tight_layout()

def export_insights(text_data, summary):
    pdf_buffer = BytesIO(export_pdf(text_data))
    buffer_txt = BytesIO(text_data.encode("utf-8"))
    buffer_json = BytesIO(json.dumps(summary, indent=4).encode("utf-8"))
    return pdf_buffer, buffer_txt, buffer_json

# API Route
@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    data = request.get_json()
    email_content = data.get("email_content", "")

    if not email_content:
        return jsonify({"error": "No email content provided"}), 400

    try:
        # Perform AI-like responses (using google.generativeai for content generation)
        summary = get_ai_response("Summarize the email in a concise, actionable format:\n\n", email_content)
        response = get_ai_response("Draft a professional response to this email:\n\n", email_content)
        highlights = get_ai_response("Highlight key points and actions in this email:\n\n", email_content)

        # Default Analysis Features
        sentiment = get_sentiment(email_content)
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        wordcloud = generate_wordcloud(email_content)
        wordcloud_fig = BytesIO()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(wordcloud_fig, format="png")
        wordcloud_fig.seek(0)

        actionable_items = extract_actionable_items(email_content)
        root_cause = detect_root_cause(email_content)
        risk = assess_risk(email_content)
        severity = detect_severity(email_content)
        critical_terms = identify_critical_keywords(email_content)
        escalation_trigger = detect_escalation_trigger(email_content)
        culprit = identify_culprit(email_content)
        tone = customer_tone_analysis(email_content)

        # Pack the results
        results = {
            "summary": summary,
            "response": response,
            "highlights": highlights,
            "sentiment": sentiment_label,
            "actionable_items": actionable_items,
            "root_cause": root_cause,
            "risk_assessment": risk,
            "severity": severity,
            "critical_keywords": critical_terms,
            "escalation_trigger": escalation_trigger,
            "culprit": culprit,
            "customer_tone": tone,
            "wordcloud_image": wordcloud_fig.getvalue().decode("utf-8")  # You may want to send it as base64 or a file URL
        }

        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
