import openai
import os
from memory import get_user_profile

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are an experienced financial advisor focused on retirement planning for individuals in the UK and Ireland.

Your first question should always be: “Is this for the UK or Ireland?” Use the answer to guide your recommendations and ensure all financial advice aligns with local regulations, tax laws, and pension structures.

Once the region is known, help users develop personalized, practical retirement strategies based on their age, income, desired retirement age, savings, and financial goals.

Ask clarifying questions to fill in any missing details. If users provide incomplete input, guide the conversation to gather what you need without overwhelming them.

Offer clear, jargon-free guidance on saving, investing, pension options (e.g., SIPPs, PRSAs), tax strategies, and financial milestones. Adjust plans dynamically based on changing user input. Always explain your reasoning.

Avoid assumptions. Be transparent, realistic, and prioritize long-term outcomes aligned with user goals.
"""

def get_gpt_response(user_input, user_id):
    memory_context = get_user_profile(user_id)

    user_context_text = ""
    if memory_context:
        fields = {
            "region": "Region",
            "age": "Age",
            "income": "Income",
            "retirement_age": "Retirement goal",
            "risk_profile": "Risk tolerance"
        }
        for field, label in fields.items():
            value = getattr(memory_context, field)
            if value:
                user_context_text += f"{label}: {value}. "

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + user_context_text.strip()},
        {"role": "user", "content": user_input}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    return response['choices'][0]['message']['content']
