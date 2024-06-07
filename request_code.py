
import streamlit as st
import requests
import numpy as np
import time

class Car:
    def __init__(self,  location, fuel, speed, acceleration, braking, steering):
        
        self.location = location
        self.fuel = fuel
        self.speed = speed
        self.acceleration = acceleration
        self.braking = braking
        self.steering = steering
    
    def get_details(self):
        return f"Location: {self.location}, Fuel: {self.fuel}, Speed: {self.speed}, Acceleration: {self.acceleration}, Braking: {self.braking}, Steering: {self.steering}"

# Simulated cars with their details
cars = [
    Car( 12.0, 0.7, 28.0, 0.2, 0.0, 0.0),
    # Add more cars as needed
]

# Streamlit UI
st.title("Cars Simulation Dashboard")

for car in cars:
   
    car_details_text = st.text(car.get_details())
    suggested_action_text = st.text("Suggested Action: ")

while True:
    for car in cars:
        # Simulate live changes in car parameters
        car.location += np.random.uniform(-1, 1)
        car.fuel -= np.random.uniform(0.01, 0.05)
        car.speed += np.random.uniform(-2, 2)
        car.acceleration += np.random.uniform(-0.1, 0.1)
        car.braking += np.random.uniform(-0.1, 0.1)
        car.steering += np.random.uniform(-0.1, 0.1)

        # Define the data for the request using updated car parameters
        data = {
            "location": car.location,
            "fuel": car.fuel,
            "speed": car.speed,
            "acceleration": car.acceleration,
            "braking": car.braking,
            "steering": car.steering
        }

        # Send a POST request to the API
        response = requests.post("http://127.0.0.1:5000/get_suggestion", json=data)

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            suggested_action = result["suggested_action"]
            
            # Update the text elements directly
            car_details_text.text(f"Car Details: {car.get_details()}")
            suggested_action_text.text(f"Suggested Action: {suggested_action}")

        # Wait for a few seconds before sending the next request
        time.sleep(1)  # Adjust the sleep time as needed
