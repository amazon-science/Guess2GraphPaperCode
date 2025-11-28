import requests
import json

def call_bedrock_with_api_key(prompt, api_key, model_id="anthropic.claude-v2", max_tokens=1000):
    """
    Call Amazon Bedrock using API key authentication
    
    Args:
        prompt (str): Input prompt text
        api_key (str): Bedrock API key
        model_id (str): Bedrock model ID to use
        max_tokens (int): Maximum tokens in response
        
    Returns:
        str: Model response text
    """
    
    # Bedrock API endpoint
    url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/invoke"
    
    # Create request body based on model
    if "claude" in model_id:
        body = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    elif "titan" in model_id:
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
    
    # Headers with API key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Make request
    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code == 200:
        response_data = response.json()
        if "claude" in model_id:
            return response_data.get('completion')
        elif "titan" in model_id:
            return response_data.get('results')[0].get('outputText')
    else:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")

# Example usage
if __name__ == "__main__":
    # TODO: Replace with your AWS Bedrock API key
    api_key = "YOUR_AWS_BEDROCK_API_KEY_HERE"
    
    try:
        prompt = "What is machine learning?"
        response = call_bedrock_with_api_key(prompt, api_key)
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this code:")
        print("1. Generate a Bedrock API key in the AWS Console")
        print("2. Replace 'YOUR_BEDROCK_API_KEY_HERE' with your actual key")