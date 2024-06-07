# run_apps.py
import subprocess
import time
import threading

# Function to run the Flask API
def run_flask():
    subprocess.run(["python", "API_code.py"])

# Start the Flask API in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# Give the Flask API some time to start
time.sleep(5)

# Start the Streamlit app
subprocess.run(["streamlit", "run", "streamlit_app.py"])
