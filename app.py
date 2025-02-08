from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import json
from fpdf import FPDF

# Initialize FastAPI app
app = FastAPI()

# Configure Google AI API Key
genai.configure(api_key="YOUR_GOOGLE_API_KEY")  # Replace with your actual API key

MAX_EMAIL_LENGTH = 1000

class EmailContent(BaseModel):
    email_text: str

# Helper Functions (revised to work dynamically with AI)
def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during AI response generation: {str(e)}")

def analyze_sentiment(email_content):
    prompt = f"Analyze the sentiment of the following email content:\n\n{email_content}\n\nIs it positive, neutral, or negative?"
    return get_ai_response(prompt, email_content)

def extract_dynamic_keywords(email_content):
    prompt = f"Identify critical keywords from the following email content:\n\n{email_content}\n\nProvide a list of the most important words or phrases."
    return get_ai_response(prompt, email_content)

def extract_actionable_items(email_content):
    prompt = f"Identify any actionable items from the following email content:\n\n{email_content}\n\nWhat are the next steps or tasks mentioned?"
    return get_ai_response(prompt, email_content)

def detect_root_cause(email_content):
    prompt = f"Based on the email, can you identify any possible root causes for the issues discussed in the email?\n\n{email_content}\n\nPlease provide your analysis."
    return get_ai_response(prompt, email_content)

def identify_potential_culprit(email_content):
    prompt = f"Based on the email, who or what might be responsible for the issues mentioned? Provide your analysis.\n\n{email_content}\n\nPlease identify any potential culprits."
    return get_ai_response(prompt, email_content)

def analyze_trends(email_content):
    prompt = f"Based on the email, identify any trends or patterns in the work, team, or issues discussed. Please analyze the content.\n\n{email_content}\n\nWhat trends or recurring issues do you detect?"
    return get_ai_response(prompt, email_content)

def assess_risk(email_content):
    prompt = f"Analyze the following email content for potential risks. Is there any concern about timelines, resources, or outcomes?\n\n{email_content}\n\nProvide a risk assessment."
    return get_ai_response(prompt, email_content)

def detect_severity(email_content):
    prompt = f"Analyze the severity of the issues discussed in the following email. Is this urgent or normal?\n\n{email_content}\n\nPlease provide a severity level."
    return get_ai_response(prompt, email_content)

# PDF export function (unchanged)
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

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
        sentiment = await analyze_sentiment(email_content)

        # Dynamic Keywords Extraction
        dynamic_keywords = await extract_dynamic_keywords(email_content)

        # Extract Actionable Items
        actionable_items = await extract_actionable_items(email_content)

        # Root Cause Detection
        root_cause = await detect_root_cause(email_content)

        # Culprit Identification
        culprit = await identify_potential_culprit(email_content)

        # Trend Analysis
        trends = await analyze_trends(email_content)

        # Risk Assessment
        risk = await assess_risk(email_content)

        # Severity Detection
        severity = await detect_severity(email_content)

        # Prepare Response Data
        response_data = {
            "summary": summary,
            "response": response,
            "highlights": highlights,
            "sentiment": sentiment,
            "dynamic_keywords": dynamic_keywords,
            "actionable_items": actionable_items,
            "root_cause": root_cause,
            "culprit": culprit,
            "trends": trends,
            "risk": risk,
            "severity": severity
        }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during email analysis: {str(e)}")

@app.post("/export-pdf")
async def export_pdf_endpoint(content: EmailContent):
    try:
        summary = get_ai_response("Summarize the email in a concise, actionable format:\n\n", content.email_text)
        response_data = f"Summary:\n{summary}\n"
        pdf_data = export_pdf(response_data)
        return {"pdf": pdf_data}
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=f"Error during JSON export: {str(e)}")

# Run the application with: uvicorn your_file_name:app --reload
