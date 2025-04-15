from openai import OpenAI
from memory import get_user_profile
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an experienced financial advisor focused on retirement planning for individuals in the UK and Ireland.

Use the user's stored information to personalize your guidance. Be warm, clear, and concise.

Avoid repeating initial questions (e.g. region) if the information is already known.

Only ask for what’s missing. Always explain your reasoning.
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

    # ✅ Exit early if init message
    if user_input.strip() == "__INIT__":
        if profile:
            name_guess = user_id.split("-")[0].capitalize()
            summary = format_user_context(profile)
            return f"Hi {name_guess}, good to see you again. Here's what I remember about your situation: {summary}"
        else:
            return (
                "Hi, glad to have time to chat with you. How can I help with your pension?"
                "\n\nTo start, could you tell me if you're based in the UK or Ireland?"
            )

    # Ongoing chat logic
    if profile:
        name_guess = user_id.split("-")[0].capitalize()
        greeting = f"Hi {name_guess}, good to see you again."
        review = format_user_context(profile)
        user_context = f"{greeting} Here's what I remember about your situation: {review}"
    else:
        user_context = "Hi, glad to have time to chat with you. How can I help with your pension?"

    # Region-specific prompt
    follow_up_note = ""
    if profile and profile.region and profile.region.lower() == "ireland":
        follow_up_note = "\n\nIf the user is in Ireland, you must ask: 'Do you know how many years of PRSI contributions you have made?'"

    if not profile or not profile.region:
        user_context += "\n\nTo start, could you tell me if you're based in the UK or Ireland?"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + user_context.strip() + follow_up_note},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content
