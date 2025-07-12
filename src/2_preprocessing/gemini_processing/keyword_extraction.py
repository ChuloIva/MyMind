import google.generativeai as genai
import os
import json

# It's recommended to set your API key as an environment variable
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# Or, for demonstration purposes, you can set it directly (not recommended for production)
api_key = "YOUR_API_KEY"

genai.configure(api_key=api_key)

def extract_keywords_with_sentiment(text):
    """
    Extracts keywords from a given text and analyzes their sentiment using the Gemini API.

    Args:
        text: The text to analyze.

    Returns:
        A list of dictionaries, where each dictionary contains a keyword and its sentiment score.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Extract the main keywords from the following text. For each keyword, provide a sentiment score from -1 (very negative) to 1 (very positive).
    Return the keywords and their sentiment scores as a JSON object with a single key "keywords" which contains a list of objects, each with a "keyword" and a "sentiment" key.

    Text: "{text}"
    """
    response = model.generate_content(prompt)
    try:
        # The response text might contain markdown, so we need to clean it
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        keywords = json.loads(cleaned_response)
        return keywords.get("keywords", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing keywords: {e}")
        return []

# --- Example Usage ---
sample_text = "The new Gemini API is incredibly powerful and easy to use. I'm very impressed with its capabilities. However, I had a really bad experience with their customer service. It was slow and unhelpful."

keywords_with_sentiment = extract_keywords_with_sentiment(sample_text)
print(f"Keywords with sentiment: {json.dumps(keywords_with_sentiment, indent=2)}")
