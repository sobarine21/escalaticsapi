from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import matplotlib.pyplot as plt
from fpdf import FPDF
import logging
import os
import base64
from io import BytesIO

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

# Word Cloud Generation
def generate_wordcloud(text):
    word_counts = {}
    for word in text.split():
        word = word.lower()
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1
    return word_counts

# Export to PDF
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

# Helper function to generate word cloud image as base64
def generate_wordcloud_image(word_counts):
    fig = plt.figure(figsize=(10, 5))
    plt.bar(word_counts.keys(), word_counts.values())
    plt.xticks(rotation=45)
    plt.title("Word Frequency")
    plt.tight_layout()

    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    # Convert the image to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Close the figure after saving
    return img_base64

# API endpoint to analyze email content
@app.post("/analyze-email")
async def analyze_email(content: EmailContent):
    email_content = content.email_text
    try:
        # Generate AI-like responses (using google.generativeai for content generation)
        model = genai.GenerativeModel("gemini-1.5-flash")
        summary = model.generate_content(f"Summarize the email in a concise, actionable format:\n\n{email_content[:MAX_EMAIL_LENGTH]}").text.strip()
        
        # Sentiment Analysis
        sentiment = get_sentiment(email_content)
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Generate Word Cloud and get it as base64 image
        word_counts = generate_wordcloud(email_content)
        wordcloud_image_base64 = generate_wordcloud_image(word_counts)

        # Prepare Response
        response_data = {
            "summary": summary,
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment
            },
            "wordcloud": wordcloud_image_base64
        }

        return response_data

    except Exception as e:
        logger.error(f"Error during email analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
