from gpt_engine import get_gpt_response
from memory import save_user_profile
from main import extract_user_data

test_user = "debug123"
test_tone = "14"

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in ["exit", "quit"]:
        break

    extract_user_data(test_user, prompt)
    reply = get_gpt_response(prompt, test_user, tone=test_tone)
    print("Pension Guru:", reply)
