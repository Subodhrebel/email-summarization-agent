import json
import requests
from datetime import date

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

SYSTEM_PROMPT = """
You are an Email Summarization Agent.

Your job:
1. Summarize the email in 2–3 sentences
2. Extract key points
3. Extract action items (who should do what)
4. Identify deadlines
5. Classify urgency: Low, Medium, or High

Return ONLY valid JSON with this schema:

{
  "summary": "",
  "key_points": [],
  "action_items": [],
  "deadlines": [],
  "urgency": ""
}
"""

def read_email(path="email.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def call_llm(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"Ollama API Error: {response.text}")

    return response.json()["response"]

def summarize_email(email_text):
    full_prompt = SYSTEM_PROMPT + "\n\nEmail:\n" + email_text
    llm_output = call_llm(full_prompt)
    return json.loads(llm_output)

def save_outputs(data):
    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Email Summary ({date.today()})\n")
        f.write("=" * 40 + "\n\n")
        f.write("SUMMARY:\n")
        f.write(data["summary"] + "\n\n")

        f.write("KEY POINTS:\n")
        for p in data["key_points"]:
            f.write(f"- {p}\n")

        f.write("\nACTION ITEMS:\n")
        for a in data["action_items"]:
            f.write(f"- {a}\n")

        f.write("\nDEADLINES:\n")
        for d in data["deadlines"]:
            f.write(f"- {d}\n")

        f.write(f"\nURGENCY: {data['urgency']}\n")

def main():
    email_text = read_email()
    result = summarize_email(email_text)
    save_outputs(result)
    print("✅ Email summarized successfully.")
    print(result)

if __name__ == "__main__":
    main()