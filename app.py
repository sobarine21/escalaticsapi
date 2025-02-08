from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import logging
import os
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

# Sentiment Analysis using Google AI
def get_sentiment(email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        sentiment_result = model.generate_content(f"Analyze the sentiment of the following text:\n\n{email_content[:MAX_EMAIL_LENGTH]}")
        sentiment_analysis = sentiment_result.strip()  # Fixed: Remove .text, directly use the string response
        return sentiment_analysis
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        return "Unknown sentiment"

# Root Cause Identification using Google AI
def identify_root_cause(email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        root_cause_result = model.generate_content(f"Identify potential root causes or issues mentioned in the following email content:\n\n{email_content[:MAX_EMAIL_LENGTH]}")
        root_cause = root_cause_result.strip()  # Fixed: Remove .text, directly use the string response
        return root_cause
    except Exception as e:
        logger.error(f"Error during root cause analysis: {str(e)}")
        return "Unknown root cause"

# Suggested Response Generation using Google AI
def generate_suggested_response(email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response_result = model.generate_content(f"Generate a suggested response to the following email:\n\n{email_content[:MAX_EMAIL_LENGTH]}")
        suggested_response = response_result.strip()  # Fixed: Remove .text, directly use the string response
        return suggested_response
    except Exception as e:
        logger.error(f"Error during response generation: {str(e)}")
        return "Unable to generate response"

# Export to PDF
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

# API endpoint to analyze email content
@app.post("/analyze-email")
async def analyze_email(content: EmailContent):
    email_content = content.email_text
    try:
        # Generate AI-like responses (using google.generativeai for content generation)
        model = genai.GenerativeModel("gemini-1.5-flash")
        summary = model.generate_content(f"Summarize the email in a concise, actionable format:\n\n{email_content[:MAX_EMAIL_LENGTH]}").strip()  # Fixed: Remove .text, directly use the string response

        # Sentiment Analysis - Use AI model to get dynamic sentiment
        sentiment = get_sentiment(email_content)

        # Root Cause Identification - Use AI model to identify issues
        root_cause = identify_root_cause(email_content)

        # Suggested Response - Generate a suggested response to the email
        suggested_response = generate_suggested_response(email_content)

        # Prepare Response
        response_data = {
            "summary": summary,
            "sentiment": sentiment,
            "root_cause": root_cause,
            "suggested_response": suggested_response
        }

        return response_data

    except Exception as e:
        logger.error(f"Error during email analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
