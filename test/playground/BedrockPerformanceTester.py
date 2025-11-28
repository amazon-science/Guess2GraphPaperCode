import os
import boto3
import time
import statistics
from datetime import datetime, timedelta

class BedrockPerformanceTester:
    def __init__(self, api_key, region_name="us-east-1"):
        self.api_key = api_key
        os.environ['AWS_BEARER_TOKEN_BEDROCK'] = api_key
        self.region = region_name
        self.client = boto3.client(service_name="bedrock-runtime", region_name=region_name)
        
        # Statistics tracking
        self.response_times = []
        self.error_counts = {"429": 0, "503": 0, "400": 0, "other": 0}
        self.successful_requests = 0
        self.total_requests = 0
        self.consecutive_failures = 0
        self.current_wait_time = 1  # Start with 10s wait time
        
    def make_request(self, prompt, model_id, max_tokens=200):
        """Make a single request and return (success, response_time, error_type, response_text)"""
        start_time = time.time()
        try:
            response = self.client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.7, "topP": 0.9}
            )
            end_time = time.time()
            response_time = end_time - start_time
            # Handle different response formats
            try:
                content = response['output']['message']['content'][0]
                if 'text' in content:
                    # Standard format (Llama, Claude, etc.)
                    response_text = content['text']
                elif 'reasoningContent' in content:
                    # OpenAI format with reasoning
                    reasoning = content['reasoningContent']
                    if isinstance(reasoning, dict) and 'reasoningText' in reasoning:
                        reasoning_text = reasoning['reasoningText']
                        if isinstance(reasoning_text, dict) and 'text' in reasoning_text:
                            response_text = reasoning_text['text']
                        else:
                            response_text = str(reasoning_text)
                    else:
                        response_text = str(reasoning)
                else:
                    response_text = str(content)
            except (KeyError, IndexError, TypeError):
                # Fallback for different response structures
                response_text = str(response.get('output', response))
            self.response_times.append(response_time)
            self.successful_requests += 1
            self.consecutive_failures = 0
            return True, response_time, None, response_text
            
        except Exception as e:
            print(f"Error details: {e}")
            end_time = time.time()
            response_time = end_time - start_time
            self.consecutive_failures += 1
            
            error_str = str(e)
            if "429" in error_str or "throttl" in error_str.lower():
                self.error_counts["429"] += 1
                return False, response_time, "429", None
            elif "503" in error_str or "unavailable" in error_str.lower():
                self.error_counts["503"] += 1
                return False, response_time, "503", None
            elif "400" in error_str or "ValidationException" in error_str:
                self.error_counts["400"] += 1
                return False, response_time, "400", None
            elif "AccessDeniedException" in error_str or "access" in error_str.lower():
                self.error_counts["other"] += 1
                return False, response_time, "access_denied", None
            else:
                self.error_counts["other"] += 1
                return False, response_time, "other", None
        finally:
            self.total_requests += 1
    
    def wait_with_adaptive_backoff(self, error_type):
        """Wait with adaptive backoff - increase by 2s on rate limit"""
        if error_type == "429":  # Rate limit - increase wait time
            self.current_wait_time += 2
            print(f"    Rate limit hit, increasing wait to {self.current_wait_time}s...")
        elif error_type == "503":  # Service unavailable
            wait_time = 5
            print(f"    Service unavailable, waiting {wait_time}s...")
            time.sleep(wait_time)
            return wait_time
        else:
            wait_time = 1
            print(f"    Other error, waiting {wait_time}s...")
            time.sleep(wait_time)
            return wait_time
        
        time.sleep(self.current_wait_time)
        return self.current_wait_time
    
    def test_baseline_response_time(self, prompt, model_id):
        """Phase 1: Test baseline response time with 10 requests"""
        print(f"\n=== Phase 1: Baseline Response Time Testing ===")
        print(f"Sending 10 requests with {self.current_wait_time}s intervals (adaptive)...")
        
        baseline_times = []
        for i in range(10):
            print(f"  Request {i+1}/10...", end=" ")
            success, response_time, error_type, response_text = self.make_request(prompt, model_id)
            
            if success:
                baseline_times.append(response_time)
                print(f"✓ {response_time:.2f}s (wait: {self.current_wait_time}s)")
                print(f"    Response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
            else:
                print(f"✗ Error: {error_type}")
                if error_type in ["429", "503"]:
                    self.wait_with_adaptive_backoff(error_type)
            
            if i < 9:  # Don't wait after last request
                time.sleep(self.current_wait_time)
        
        if baseline_times:
            avg_time = statistics.mean(baseline_times)
            print(f"  Baseline average response time: {avg_time:.2f}s")
            print(f"  Final wait time discovered: {self.current_wait_time}s")
            return avg_time
        else:
            print(f"  No successful baseline requests!")
            return None
    
    def test_rate_limits(self, prompt, model_id, test_duration_minutes):
        """Phase 2: Discover rate limits starting from 10 RPM"""
        print(f"\n=== Phase 2: Rate Limit Discovery ===")
        
        current_rpm = 10
        max_successful_rpm = 0
        
        end_time = datetime.now() + timedelta(minutes=test_duration_minutes * 0.4)  # Use 40% of time for this phase
        
        while datetime.now() < end_time and current_rpm <= 120:  # Cap at 120 RPM
            print(f"\nTesting {current_rpm} requests per minute...")
            interval = 60.0 / current_rpm
            
            # Test this rate for 1 minute or until 2 consecutive failures
            test_start = time.time()
            requests_sent = 0
            failures_in_window = 0
            
            while time.time() - test_start < 60 and self.consecutive_failures < 2:
                success, response_time, error_type, response_text = self.make_request(prompt, model_id)
                requests_sent += 1
                
                if success:
                    print(f"  ✓ Request {requests_sent}: {response_time:.2f}s")
                    print(f"    Response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                    failures_in_window = 0
                else:
                    failures_in_window += 1
                    print(f"  ✗ Request {requests_sent}: {error_type}")
                    if error_type in ["429", "503"]:
                        self.wait_with_adaptive_backoff(error_type)
                
                # Wait for next request
                time.sleep(interval)
            
            if self.consecutive_failures < 2:
                max_successful_rpm = current_rpm
                print(f"  ✓ {current_rpm} RPM successful")
                current_rpm = min(120, int(current_rpm * 1.5))  # Increase by 50%
            else:
                print(f"  ✗ {current_rpm} RPM failed (2 consecutive errors)")
                break
        
        print(f"  Maximum successful rate: {max_successful_rpm} RPM")
        return max_successful_rpm
    
    def test_cooldown_period(self, prompt, model_id):
        """Phase 3: Find minimum cooldown period"""
        print(f"\n=== Phase 3: Cooldown Period Testing ===")
        
        intervals = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]  # Test intervals in seconds, starting from 10s
        min_safe_interval = 10.0
        
        for interval in intervals:
            print(f"\nTesting {interval}s interval...")
            self.consecutive_failures = 0
            
            # Send 5 requests at this interval
            for i in range(5):
                success, response_time, error_type, response_text = self.make_request(prompt, model_id)
                
                if success:
                    print(f"  ✓ Request {i+1}: {response_time:.2f}s")
                    print(f"    Response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                else:
                    print(f"  ✗ Request {i+1}: {error_type}")
                    break
                
                if i < 4:  # Don't wait after last request
                    time.sleep(interval)
            
            if self.consecutive_failures == 0:
                min_safe_interval = interval
                print(f"  ✓ {interval}s interval successful")
            else:
                print(f"  ✗ {interval}s interval failed")
                break
        
        print(f"  Minimum safe interval: {min_safe_interval}s")
        return min_safe_interval
    
    def sustained_load_test(self, prompt, model_id, max_rpm, remaining_time_minutes):
        """Phase 4: Sustained load test at discovered limits"""
        if remaining_time_minutes <= 0 or max_rpm == 0:
            return
            
        print(f"\n=== Phase 4: Sustained Load Test ===")
        print(f"Running at {max_rpm} RPM for {remaining_time_minutes:.1f} minutes...")
        
        interval = 60.0 / max_rpm
        end_time = datetime.now() + timedelta(minutes=remaining_time_minutes)
        
        while datetime.now() < end_time:
            success, response_time, error_type, response_text = self.make_request(prompt, model_id)
            
            if success:
                print(f"  ✓ {response_time:.2f}s")
                print(f"    Response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
            else:
                print(f"  ✗ {error_type}")
                if error_type in ["429", "503"]:
                    self.wait_with_adaptive_backoff(error_type)
            
            time.sleep(interval)
    
    def generate_final_report(self, model_id, max_rpm, min_interval):
        """Generate comprehensive final report"""
        print(f"\n" + "="*80)
        print(f"FINAL PERFORMANCE REPORT")
        print(f"="*80)
        print(f"Model: {model_id}")
        print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nREQUEST STATISTICS:")
        print(f"  Total requests: {self.total_requests}")
        print(f"  Successful requests: {self.successful_requests}")
        print(f"  Success rate: {(self.successful_requests/self.total_requests*100):.1f}%")
        
        if self.response_times:
            print(f"\nRESPONSE TIME ANALYSIS:")
            print(f"  Average: {statistics.mean(self.response_times):.2f}s")
            print(f"  Median: {statistics.median(self.response_times):.2f}s")
            print(f"  Min: {min(self.response_times):.2f}s")
            print(f"  Max: {max(self.response_times):.2f}s")
            if len(self.response_times) > 1:
                print(f"  Std Dev: {statistics.stdev(self.response_times):.2f}s")
        
        print(f"\nRATE LIMITS:")
        print(f"  Maximum RPM: {max_rpm}")
        print(f"  Minimum cooldown: {min_interval}s")
        
        print(f"\nERROR BREAKDOWN:")
        for error_type, count in self.error_counts.items():
            if count > 0:
                print(f"  {error_type}: {count} errors")
        
        print(f"="*80)

def main():
    # Available models from your Bedrock access
    models = {
        "1": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "3": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "4": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "5": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "6": "anthropic.claude-3-opus-20240229-v1:0",
        "7": "anthropic.claude-3-sonnet-20240229-v1:0",
        "8": "anthropic.claude-3-haiku-20240307-v1:0",
        "9": "anthropic.claude-v2:1",
        "10": "anthropic.claude-v2",
        "11": "anthropic.claude-instant-v1",
        "12": "meta.llama3-8b-instruct-v1:0",
        "13": "meta.llama3-70b-instruct-v1:0",
        "14": "us.meta.llama3-1-8b-instruct-v1:0",
        "15": "us.meta.llama3-1-70b-instruct-v1:0",
        "16": "us.meta.llama3-2-1b-instruct-v1:0",
        "17": "us.meta.llama3-2-3b-instruct-v1:0",
        "18": "us.meta.llama3-3-70b-instruct-v1:0",
        "19": "us.meta.llama4-scout-17b-instruct-v1:0",
        "20": "us.meta.llama4-maverick-17b-instruct-v1:0",
        "21": "amazon.titan-text-lite-v1",
        "22": "amazon.titan-text-express-v1",
        "23": "cohere.command-r-plus-v1:0",
        "24": "cohere.command-r-v1:0",
        "25": "mistral.mistral-7b-instruct-v0:2",
        "26": "openai.gpt-oss-20b-1:0",
        "27": "openai.gpt-oss-120b-1:0",
    }
    
    model_names = {
        "1": "Claude 3.5 Sonnet v2 (inference profile, optimized)",
        "2": "Claude 3.5 Sonnet v2 (direct, newest)",
        "3": "Claude 3.5 Sonnet v1 (inference profile)",
        "4": "Claude 3.5 Sonnet v1 (direct)",
        "5": "Claude 3.5 Haiku (fast, efficient)",
        "6": "Claude 3 Opus (most powerful)",
        "7": "Claude 3 Sonnet (balanced)",
        "8": "Claude 3 Haiku (fast)",
        "9": "Claude v2.1 (legacy, capable)",
        "10": "Claude v2 (legacy)",
        "11": "Claude Instant (legacy, fast)",
        "12": "Llama 3 8B Instruct (fast, cost-effective)",
        "13": "Llama 3 70B Instruct (powerful, balanced)",
        "14": "Llama 3.1 8B Instruct (latest small)",
        "15": "Llama 3.1 70B Instruct (latest large)",
        "16": "Llama 3.2 1B Instruct (ultra-fast)",
        "17": "Llama 3.2 3B Instruct (small, efficient)",
        "18": "Llama 3.3 70B Instruct (newest)",
        "19": "Llama 4 Scout 17B (newest, experimental)",
        "20": "Llama 4 Maverick 17B (newest, experimental)",
        "21": "Amazon Titan Text Lite (AWS native, fast)",
        "22": "Amazon Titan Text Express (AWS native, capable)",
        "23": "Cohere Command R+ (most capable)",
        "24": "Cohere Command R (balanced)",
        "25": "Mistral 7B Instruct (efficient)",
        "26": "OpenAI GPT OSS 20B (us-west-2 only)",
        "27": "OpenAI GPT OSS 120B (us-west-2 only)",
    }
    
    print("Bedrock Performance Tester")
    print("="*50)
    
    # Get user inputs
    print("Available models:")
    for key, name in model_names.items():
        print(f"  {key}. {name}")
    print("  28. Enter custom model ID")
    
    model_choice = input("\nSelect model (1-28): ")
    
    if model_choice == "28":
        selected_model = input("Enter model ID: ")
        print(f"Selected: {selected_model}")
    elif model_choice in models:
        selected_model = models[model_choice]
        print(f"Selected: {model_names[model_choice]}")
    else:
        print("Invalid choice!")
        return
    
    prompt = input("\nEnter test prompt: ")
    test_duration = float(input("Test duration (minutes): "))
    
    # Choose region
    print("\nSelect AWS region:")
    print("  1. us-east-1 (N. Virginia) - Most models")
    print("  2. us-west-2 (Oregon) - Includes OpenAI models")
    
    region_choice = input("Select region (1-2): ")
    if region_choice == "2":
        region = "us-west-2"
        print("Selected: us-west-2 (Oregon) - OpenAI models available")
    else:
        region = "us-east-1"
        print("Selected: us-east-1 (N. Virginia)")
    
    # Initialize tester
    # TODO: Replace with your AWS Bedrock API key
    api_key = "YOUR_AWS_BEDROCK_API_KEY_HERE"
    tester = BedrockPerformanceTester(api_key, region)
    
    print(f"\nStarting performance test...")
    print(f"Model: {selected_model}")
    print(f"Prompt: {prompt}")
    print(f"Duration: {test_duration} minutes")
    
    start_time = datetime.now()
    
    # Phase 1: Baseline testing
    baseline_time = tester.test_baseline_response_time(prompt, selected_model)
    
    # Phase 2: Rate limit discovery
    max_rpm = tester.test_rate_limits(prompt, selected_model, test_duration)
    
    # Phase 3: Cooldown testing
    min_interval = tester.test_cooldown_period(prompt, selected_model)
    
    # Phase 4: Sustained load test
    elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
    remaining_time = test_duration - elapsed_minutes
    tester.sustained_load_test(prompt, selected_model, max_rpm, remaining_time)
    
    # Final report
    tester.generate_final_report(selected_model, max_rpm, min_interval)

if __name__ == "__main__":
    main()