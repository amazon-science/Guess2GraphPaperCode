import os
import boto3

def call_bedrock_with_api_key(prompt, api_key, model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0", max_tokens=1000):
    """
    Call Amazon Bedrock using API key with boto3 (following official documentation)
    
    Args:
        prompt (str): Input prompt text
        api_key (str): Bedrock API key
        model_id (str): Bedrock model ID to use
        max_tokens (int): Maximum tokens in response
        
    Returns:
        str: Model response text
    """
    
    # Set the API key as environment variable
    os.environ['AWS_BEARER_TOKEN_BEDROCK'] = api_key
    
    # Create an Amazon Bedrock client
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )
    
    # Define the messages
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    
    # Make the request using converse API
    response = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": 0.7,
            "topP": 0.9
        }
    )
    
    # Extract the response text
    return response['output']['message']['content'][0]['text']

# Example usage
if __name__ == "__main__":
    # TODO: Replace with your AWS Bedrock API key
    api_key = "YOUR_AWS_BEDROCK_API_KEY_HERE"
    
    try:
        prompt = "What is machine learning?"
        response = call_bedrock_with_api_key(prompt, api_key)
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")