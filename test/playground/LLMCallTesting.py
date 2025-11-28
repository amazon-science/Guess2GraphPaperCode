
# This file shows examples of how to call LLMs in Bedrock.

print(5)
import boto3
import json

# Initialize Bedrock client with region
# Note: This requires AWS credentials to be configured
try:
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    print("Bedrock client initialized successfully")
except Exception as e:
    print(f"Failed to initialize Bedrock client: {e}")
    bedrock = None

def call_bedrock_llm(prompt, model_id="anthropic.claude-3-haiku-20240307-v1:0", max_tokens=1000):
    """
    Call Amazon Bedrock LLM with a prompt
    
    Args:
        prompt (str): Input prompt text
        model_id (str): Bedrock model ID to use
        max_tokens (int): Maximum tokens in response
        
    Returns:
        str: Model response text
    """
    
    if bedrock is None:
        raise Exception("Bedrock client not initialized. Check AWS credentials and configuration.")
    
    # Create request body based on model
    if "claude-3" in model_id or "claude-v2" in model_id:
        if "claude-3" in model_id:
            # Claude 3+ uses messages format
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        else:
            # Claude v2 uses prompt format
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
    
    # Invoke model
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType='application/json'
    )
    
    # Parse response
    response_body = json.loads(response.get('body').read())
    if "claude-3" in model_id:
        return response_body.get('content')[0].get('text')
    elif "claude-v2" in model_id:
        return response_body.get('completion')
    elif "titan" in model_id:
        return response_body.get('results')[0].get('outputText')

# Example usage
if __name__ == "__main__":
    try:
        prompt = "What is machine learning?"
        response = call_bedrock_llm(prompt)
        print(response)
    except Exception as e:
        print(f"Error: {e}")






