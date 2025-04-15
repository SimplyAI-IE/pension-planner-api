from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from gpt_engine import get_gpt_response
from memory import save_user_profile
from models import init_db
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from models import User, SessionLocal
import re

load_dotenv()
app = FastAPI()

# Allow CORS so frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB table(s)
init_db()

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    extract_user_data(req.user_id, req.message)
    reply = get_gpt_response(req.message, req.user_id)
    return {"response": reply}

def extract_user_data(user_id, msg):
    if "ireland" in msg.lower():
        save_user_profile(user_id, "region", "Ireland")
    elif "uk" in msg.lower():
        save_user_profile(user_id, "region", "UK")

    age_match = re.search(r"\b(\d{2})\b.*old", msg.lower())
    if age_match:
        save_user_profile(user_id, "age", int(age_match.group(1)))

    income_match = re.search(r"\b(€|£)?(\d{2,6})[kK]?\b", msg.replace(",", ""))
    if income_match:
        save_user_profile(user_id, "income", int(income_match.group(2)))

    retirement_match = re.search(r"retire.*?(\d{2})", msg.lower())
    if retirement_match:
        save_user_profile(user_id, "retirement_age", int(retirement_match.group(1)))

    if "risk" in msg.lower():
        if "low" in msg.lower():
            save_user_profile(user_id, "risk_profile", "Low")
        elif "high" in msg.lower():
            save_user_profile(user_id, "risk_profile", "High")
        elif "medium" in msg.lower():
            save_user_profile(user_id, "risk_profile", "Medium")
            
@app.post("/auth/google")
async def auth_google(user_data: dict):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_data["sub"]).first()

    if not user:
        user = User(
            id=user_data["sub"],
            name=user_data["name"],
            email=user_data["email"]
        )
        db.add(user)
        db.commit()

    db.close()
    return {"status": "ok", "user_id": user_data["sub"]}