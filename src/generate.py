import requests
import os

def generate_text(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    headers = {'Content-Type': 'application/json'}
    
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    response = requests.post(url, headers=headers, json=data)

    # Print the entire response for debugging
    print("API Response Status Code:", response.status_code)
    print("API Response Body:", response.text)
    
    # Check for valid JSON response
    try:
        response_json = response.json()
    except ValueError:
        return "Invalid JSON response received."

    # Adjusted to navigate the correct structure of the response
    candidates = response_json.get("candidates", [])
    if candidates and len(candidates) > 0:
        return candidates[0].get("content", {}).get("parts", [])[0].get("text", "")
    else:
        return "No valid response received."
