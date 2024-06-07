# run_apps.py
import subprocess
import time

# Start the Flask API
api_process = subprocess.Popen(["python", "API_code.py"])

# Give the Flask API some time to start
time.sleep(5)

# Start the Streamlit app
streamlit_process = subprocess.Popen(["streamlit", "run", "request_code.py"])

# Wait for both processes to complete (this will run indefinitely)
try:
    api_process.wait()
    streamlit_process.wait()
except KeyboardInterrupt:
    # Handle cleanup if needed
    api_process.terminate()
    streamlit_process.terminate()
