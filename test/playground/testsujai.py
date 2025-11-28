import os
import boto3
                        
# TODO: Replace with your AWS Bedrock API key
# If you already set the API key as an environment variable, you can comment this line out
os.environ['AWS_BEARER_TOKEN_BEDROCK'] = "YOUR_AWS_BEDROCK_API_KEY_HERE"

# Create an Amazon Bedrock client
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1" # If you've configured a default region, you can omit this line
)

# Define the model and message
model_id = "amazon.titan-text-express-v1"
messages = [{"role": "user", "content": [{"text": "Hello"}]}]

response = client.converse(
    modelId=model_id,
    messages=messages,
)

print("Response:", response['output']['message']['content'][0]['text'])