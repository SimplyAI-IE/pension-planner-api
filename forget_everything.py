import requests

user_id = "debug123"

res = requests.post("http://localhost:8000/chat/forget", json={"user_id": user_id})
print("✅ Forget result:", res.status_code, res.text)
