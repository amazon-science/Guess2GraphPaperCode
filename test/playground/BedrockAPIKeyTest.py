import os
import boto3
import time

def call_bedrock_with_retry(prompt, api_key, model_id="anthropic.claude-3-haiku-20240307-v1:0", max_tokens=200, max_retries=10):
    """
    Call Amazon Bedrock with adaptive retry logic
    
    Args:
        prompt (str): Input prompt text
        api_key (str): Bedrock API key
        model_id (str): Bedrock model ID to use
        max_tokens (int): Maximum tokens in response
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        tuple: (response_text, wait_time_used, attempts_made)
    """
    
    # Set the API key as environment variable
    os.environ['AWS_BEARER_TOKEN_BEDROCK'] = api_key
    
    # Create an Amazon Bedrock client
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )
    
    # Define the messages for invoke_model API
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    wait_time = 2  # Only used for retries
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:  # Wait before retry attempts
                print(f"  Waiting {wait_time}s before attempt {attempt + 1}...")
                time.sleep(wait_time)
            
            # Make the request using invoke_model API
            import json
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "messages": messages
            })
            
            response = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            # Extract the response text
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text'], wait_time, attempt + 1
            
        except Exception as e:
            if "429" in str(e) or "throttl" in str(e).lower():
                wait_time += 2
                print(f"  Rate limit hit (attempt {attempt + 1}), increasing wait to {wait_time}s")
                if attempt == max_retries:
                    raise Exception(f"Failed after {max_retries + 1} attempts. Final wait time: {wait_time}s")
            else:
                raise e
    
    raise Exception(f"Max retries ({max_retries}) exceeded")

# Example usage
if __name__ == "__main__":
    # TODO: Replace with your AWS Bedrock API key
    api_key = "YOUR_AWS_BEDROCK_API_KEY_HERE"
    
    # Available Claude models
    claude_models = {
        "1": "anthropic.claude-3-haiku-20240307-v1:0",
        "2": "anthropic.claude-3-sonnet-20240229-v1:0", 
        "3": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "4": "us.anthropic.claude-3-opus-20240229-v1:0",
        "5": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "6": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "7": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "8": "us.anthropic.claude-opus-4-20250514-v1:0",
        "9": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "10": "us.anthropic.claude-opus-4-1-20250805-v1:0"
    }
    
    model_names = {
        "1": "Claude 3 Haiku (fast, cost-effective)",
        "2": "Claude 3 Sonnet (balanced)",
        "3": "Claude 3.5 Sonnet (more capable)",
        "4": "Claude 3 Opus (most capable)",
        "5": "Claude 3.5 Haiku (inference profile)",
        "6": "Claude 3.5 Sonnet (inference profile)",
        "7": "Claude 3.7 Sonnet (inference profile)",
        "8": "Claude Opus 4 (latest, most advanced)",
        "9": "Claude Sonnet 4 (latest, balanced)",
        "10": "Claude Opus 4.1 (newest, most advanced)"
    }
    
    print("Available options:")
    for key, name in model_names.items():
        print(f"{key}. {name}")
    print("11. Test ALL models with the same prompt")
    
    choice = input("\nChoose an option (1-11): ")
    
    if choice == "11":
        prompt = input("\nEnter your question to test on ALL Claude models: ")
        print("\n" + "="*80)
        print(f"TESTING PROMPT: {prompt}")
        print("="*80)
        
        for key, model_id in claude_models.items():
            print(f"\n--- {model_names[key]} ---")
            try:
                response, wait_time, attempts = call_bedrock_with_retry(prompt, api_key, model_id)
                print(f"✓ Success after {attempts} attempts (final wait: {wait_time}s)")
                print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            except Exception as e:
                print(f"✗ Failed: {e}")
            print("-" * 50)
                
    elif choice in claude_models:
        selected_model = claude_models[choice]
        print(f"Selected: {model_names[choice]}")
        
        prompt = input("\nEnter your question: ")
        
        try:
            response, wait_time, attempts = call_bedrock_with_retry(prompt, api_key, selected_model)
            print(f"\n✓ Success after {attempts} attempts (final wait: {wait_time}s)")
            print("\nResponse:", response)
        except Exception as e:
            print(f"✗ Failed: {e}")
    else:
        print("Invalid choice. Please select 1-11.")