"""
Run all tests in the test folder.
"""

import subprocess
import sys
import os


def run_test_file(test_file):
    """Run a single test file and return success status and error info"""
    print(f"\n{'='*50}")
    print(f"Running {test_file}")
    print('='*50)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, 
                              check=True,
                              text=True)
        print(f"✓ {test_file} passed")
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"✗ {test_file} failed")
        error_info = {
            'returncode': e.returncode,
            'stdout': e.stdout,
            'stderr': e.stderr
        }
        return False, error_info


if __name__ == "__main__":
    # Get test directory path
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of test files to run
    test_files = [
        "test_data_generation.py",
        "test_methods.py",
        "test_metrics.py",
        "test_experiment_baselines.py"
    ]
    
    print("Running all tests...")
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            success, error_info = run_test_file(test_path)
            if success:
                passed += 1
            else:
                failed += 1
                failed_tests.append((test_file, error_info))
        else:
            print(f"Warning: {test_file} not found")
            failed += 1
            failed_tests.append((test_file, {'error': 'File not found'}))
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {failed} test file(s) failed")
        
        # Display detailed error information
        print("\n" + "="*50)
        print("FAILED TEST DETAILS")
        print("="*50)
        
        for test_file, error_info in failed_tests:
            print(f"\n❌ {test_file}:")
            if 'error' in error_info:
                print(f"   Error: {error_info['error']}")
            else:
                print(f"   Return code: {error_info['returncode']}")
                if error_info['stderr']:
                    print(f"   Error output:\n{error_info['stderr']}")
                if error_info['stdout']:
                    print(f"   Standard output:\n{error_info['stdout']}")
        
        sys.exit(1)