import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from app.core.schemas import RoutingDecision

print("Loading Models...")
# We still use Vectors for the best "Simple" detection
embedder = SentenceTransformer('all-MiniLM-L6-v2')

simple_anchors = [
    "I forgot my password", "reset password", "login issue", "cannot sign in",
    "send me the invoice", "billing error", "refund request", "cancel subscription",
    "how do I update my account", "payment failed", "where is my receipt"
]
simple_embeddings = embedder.encode(simple_anchors, convert_to_tensor=True)

def route_complaint(text: str) -> RoutingDecision:
    # 1. Check if it is clearly Simple (Vector Match)
    user_embedding = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(user_embedding, simple_embeddings)
    best_score = torch.max(scores).item()
    
    # 2. Check Simple Keywords (Backup)
    simple_keywords = ["invoice", "password", "billing", "refund", "subscription", "login"]
    keyword_match = any(word in text.lower() for word in simple_keywords)

    # LOGIC: If it looks simple, handle it here.
    # EVERYTHING ELSE goes to the GPU (Mistral) to decide.
    if best_score > 0.35 or keyword_match:
        return RoutingDecision(
            decision="Simple",
            confidence=round(best_score, 2),
            tags=["Billing/Account"],
            reason="Detected simple administrative query"
        )

    # Default to Complex -> Let Mistral decide if it's Sarcasm or Valid
    return RoutingDecision(
        decision="Complex",
        confidence=0.0, # Irrelevant now
        tags=["Technical/Hardware"],
        reason="Routing to Tier 1b for Deep Analysis & Validation"
    )