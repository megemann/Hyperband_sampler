#!/usr/bin/env python
"""
Run all Hyperband tests.
"""
import subprocess
import sys

def run_test_file(test_file):
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"PASSED: {test_file}")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"FAILED: {test_file}")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            if result.stderr:
                print("Error:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"FAILED: {test_file} with exception: {e}")
        return False

def main():
    """Run all tests."""
    print("Running All Hyperband Tests...")
    
    test_files = [
        "test_comprehensive.py",
        "test_visualization_debug.py", 
        "test_multi_objective_sampler.py",
        "test_multi_objective_study.py"
    ]
    
    results = []
    for test_file in test_files:
        success = run_test_file(test_file)
        results.append((test_file, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for test_file, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_file}: {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print(f"\n{failed} TEST(S) FAILED!")
        return 1

if __name__ == "__main__":
    exit(main()) 