import os
import requests

def get_embedding(text: str) -> list:
    """Get embedding from Mistral API for a single text."""
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise ValueError("Set the MISTRAL_API_KEY environment variable.")
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"input": text, "model": "mistral-embed"}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    resp_json = response.json()
    try:
        return resp_json["data"][0]["embedding"]
    except Exception as e:
        print("API response did not contain expected embedding structure:", resp_json)
        raise

