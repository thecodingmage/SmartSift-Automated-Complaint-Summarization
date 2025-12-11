import os
from groq import Groq
import json
from dotenv import load_dotenv
from app.core.schemas import DetailedAnalysis

# Load API Key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ... (Keep imports and generate_executive_report) ...

def analyze_complex_complaint(text: str, complaint_id: str) -> DetailedAnalysis:
    """
    Tier 1b: Performs ABSA *AND* Sarcasm Detection.
    """
    
    # NEW PROMPT: The "Smart Judge" Logic
    system_prompt = """
    You are an expert QA Analyst. Analyze the complaint.
    
    STEP 1: Check for Sarcasm, Ambiguity, or Gibberish.
    - If the text is sarcastic (e.g., "Great job breaking it"), ambiguous (e.g., "I don't know"), or lacks clear technical details, reject it.
    - Set "status" to "Review_Queue" and "flag_reason" to explain why.
    
    STEP 2: If the complaint is valid/technical:
    - Perform Aspect-Based Sentiment Analysis.
    - Set "status" to "Success".
    
    OUTPUT JSON FORMAT:
    {
        "complaint_id": "...",
        "status": "Success" OR "Review_Queue",
        "flag_reason": "...",
        "aspects": [ {"aspect": "Battery", "sentiment": "Negative", "severity": "High"} ],
        "summary": "..."
    }
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Complaint ID: {complaint_id}\nText: {text}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        raw_json = completion.choices[0].message.content
        data = json.loads(raw_json)
        
        # Pydantic validation handles the structure
        return DetailedAnalysis(**data)

    except Exception as e:
        print(f"LLM Error: {e}")
        return None
    


# ... (Keep existing imports and analyze_complex_complaint function above) ...

def generate_executive_report(stats: dict) -> str:
    """
    Sends aggregated stats to Groq (Llama 3.3) for a strategic summary.
    (Replaces Claude to save costs/time)
    """
    # Reuse the existing Groq client (client) we set up at the top of this file
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not found in .env"

    # 1. THE PROMPT
    system_prompt = f"""
    You are a Senior Product Strategy Executive. 
    Analyze the following defect statistics from our user complaints.
    
    DATA:
    {json.dumps(stats, indent=2)}
    
    OUTPUT REQUIREMENTS:
    1. Identify the top critical risk.
    2. Propose 3 specific engineering actions.
    3. Estimate the business impact if ignored.
    4. Keep it professional, concise, and action-oriented.
    """

    try:
        # We use the same smart model (Llama 3.3) as the analysis engine
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the Executive Strategy Report now."}
            ],
            temperature=0.3
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error generating report: {e}"