#!/usr/bin/env python3
"""
Test script to verify Streamlit app can start without errors
"""

import subprocess
import time
import sys
import signal
import os

def test_streamlit_startup():
    """Test if Streamlit app starts without import/syntax errors"""
    print("🧪 Testing Streamlit application startup...")
    
    # Start Streamlit in headless mode
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.headless", "true",
        "--server.port", "8502",  # Use different port to avoid conflicts
        "--server.runOnSave", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        # Start the process
        print("🚀 Starting Streamlit server...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # Wait a few seconds for startup
        startup_time = 10
        print(f"⏳ Waiting {startup_time} seconds for server to start...")
        
        for i in range(startup_time):
            time.sleep(1)
            if process.poll() is not None:
                # Process has terminated
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    print(f"❌ Streamlit failed to start!")
                    print(f"Error output: {stderr}")
                    return False
                break
            print(f"   ... {i+1}/{startup_time}")
        
        # If we get here, the process is still running
        if process.poll() is None:
            print("✅ Streamlit server started successfully!")
            
            # Try to get some output
            try:
                stdout, stderr = process.communicate(timeout=2)
                if "You can now view your Streamlit app" in stdout:
                    print("✅ App is accessible!")
                if stderr and "error" in stderr.lower():
                    print(f"⚠️  Some warnings/errors: {stderr[:200]}...")
            except subprocess.TimeoutExpired:
                # This is actually good - server is running
                print("✅ Server is running (timeout on communication is expected)")
            
            # Terminate the process
            try:
                process.terminate()
                process.wait(timeout=5)
                print("✅ Server stopped cleanly")
            except subprocess.TimeoutExpired:
                process.kill()
                print("⚠️  Had to force kill the server")
            
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Process terminated unexpectedly!")
            print(f"Return code: {process.returncode}")
            print(f"Output: {stdout}")
            print(f"Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Streamlit startup: {e}")
        return False

if __name__ == "__main__":
    print("🌳 Tree Health Monitor - Streamlit Startup Test\n")
    
    if test_streamlit_startup():
        print("\n🎉 Streamlit application startup test PASSED!")
        print("\n💡 The application is ready to use!")
        print("   Run: streamlit run streamlit_app.py")
        print("   Or:  .\\run_streamlit.ps1")
    else:
        print("\n❌ Streamlit application startup test FAILED!")
        print("   Please check the error messages above.")