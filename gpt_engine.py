from openai import OpenAI
from memory import get_user_profile
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an experienced financial advisor focused on retirement planning for individuals in the UK and Ireland.

Your first question should always be: “Is this for the UK or Ireland?” Use the answer to guide your recommendations and ensure all financial advice aligns with local regulations, tax laws, and pension structures.

Once the region is known, help users develop personalized, practical retirement strategies based on their age, income, desired retirement age, savings, and financial goals.

Ask clarifying questions to fill in any missing details. If users provide incomplete input, guide the conversation to gather what you need without overwhelming them.

Offer clear, jargon-free guidance on saving, investing, pension options (e.g., SIPPs, PRSAs), tax strategies, and financial milestones. Adjust plans dynamically based on changing user input. Always explain your reasoning.

Avoid assumptions. Be transparent, realistic, and prioritize long-term outcomes aligned with user goals.
"""

def format_user_context(profile):
    parts = []
    if profile.region:
        parts.append(f"Region: {profile.region}")
    if profile.age:
        parts.append(f"Age: {profile.age}")
    if profile.income:
        parts.append(f"Income: £{profile.income}")
    if profile.retirement_age:
        parts.append(f"Retirement goal: {profile.retirement_age}")
    if profile.risk_profile:
        parts.append(f"Risk tolerance: {profile.risk_profile}")
    return " ".join(parts)

def get_gpt_response(user_input, user_id):
    profile = get_user_profile(user_id)
    user_context = format_user_context(profile) if profile else ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + user_context.strip()},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content
