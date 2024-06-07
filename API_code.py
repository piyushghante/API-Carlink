
from flask import Flask, request, jsonify
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical



app = Flask(__name__)

# Function to preprocess the input data
def preprocess_input(location, fuel, speed, acceleration, braking, steering):
    normalized_location = location / max_location
    normalized_fuel = fuel / max_fuel
    normalized_speed = speed / max_speed
    normalized_acceleration = acceleration / max_acceleration
    normalized_braking = braking / max_braking
    normalized_steering = steering / max_steering
    return np.array([normalized_location, normalized_fuel, normalized_speed, normalized_acceleration, normalized_braking, normalized_steering, 0.0])

# Function to perform the traffic control action based on the current state
def control_traffic(location, fuel, speed, acceleration, braking, steering):
    state = preprocess_input(location, fuel, speed, acceleration, braking, steering)
    action_probs = model.predict(np.array([state]))[0]
    action = np.argmax(action_probs)
    if action == 0:
        return "accelerate"
    elif action == 1:
        return "brake"
    elif action == 2:
        return "maintain_speed"

class Car:
    def __init__(self, location, fuel, speed, acceleration, braking, steering):
        self.location = location
        self.fuel = fuel
        self.speed = speed
        self.acceleration = acceleration
        self.braking = braking
        self.steering = steering

    def get_details(self):
        return f"Location: {self.location}, Fuel: {self.fuel}, Speed: {self.speed}, Acceleration: {self.acceleration}, Braking: {self.braking}, Steering: {self.steering}"

# Training data and normalization constants
training_data = [
    [10.0, 0.8, 30.0, 0.5, 0.0, 0.0, 0],  
    [20.0, 0.7, 40.0, 0.0, 0.2, 0.0, 1],  
    [15.0, 0.6, 35.0, 0.0, 0.0, 0.5, 2],  
    [8.0, 0.7, 28.0, 0.4, 0.0, 0.1, 0],  
    [22.0, 0.6, 40.0, 0.0, 0.4, 0.0, 1],  
    [17.0, 0.5, 35.0, 0.0, 0.0, 0.3, 2],  
    [9.0, 0.4, 25.0, 0.2, 0.0, 0.2, 0],  
    [11.0, 0.6, 32.0, 0.0, 0.3, 0.0, 1],  
    [19.0, 0.8, 43.0, 0.0, 0.0, 0.0, 2],  
    [6.0, 0.3, 22.0, 0.3, 0.0, 0.4, 0],  
    [24.0, 0.5, 36.0, 0.0, 0.5, 0.0, 1],  
    [14.0, 0.4, 29.0, 0.0, 0.0, 0.5, 2],  
    [7.0, 0.6, 27.0, 0.5, 0.0, 0.0, 0],  
    [23.0, 0.7, 42.0, 0.0, 0.3, 0.2, 1],  
    [16.0, 0.5, 34.0, 0.0, 0.0, 0.4, 2],  
    [10.0, 0.3, 23.0, 0.4, 0.0, 0.3, 0],  
    [21.0, 0.6, 39.0, 0.0, 0.4, 0.0, 1],  
    [18.0, 0.4, 36.0, 0.0, 0.0, 0.5, 2],  
    [8.0, 0.6, 26.0, 0.5, 0.0, 0.0, 0],  
    [25.0, 0.7, 41.0, 0.0, 0.3, 0.1, 1],  
    [15.0, 0.5, 33.0, 0.0, 0.0, 0.4, 2],  
    [9.0, 0.3, 24.0, 0.4, 0.0, 0.2, 0],  
    [20.0, 0.5, 38.0, 0.0, 0.4, 0.0, 1],  
    [17.0, 0.4, 35.0, 0.0, 0.0, 0.5, 2],  
    [7.0, 0.5, 27.0, 0.5, 0.0, 0.0, 0],  
    [23.0, 0.6, 40.0, 0.0, 0.3, 0.2, 1],  
    [14.0, 0.5, 32.0, 0.0, 0.0, 0.4, 2],  
    [10.0, 0.4, 25.0, 0.4, 0.0, 0.3, 0],  
    [21.0, 0.7, 38.0, 0.0, 0.4, 0.0, 1],  
    [16.0, 0.6, 34.0, 0.0, 0.0, 0.5, 2],  
    [8.0, 0.5, 26.0, 0.5, 0.0, 0.0, 0],  
    [24.0, 0.6, 40.0, 0.0, 0.3, 0.2, 1],  
    [15.0, 0.4, 33.0, 0.0, 0.0, 0.4, 2],  
    [9.0, 0.3, 24.0, 0.4, 0.0, 0.3, 0],  
    [20.0, 0.6, 37.0, 0.0, 0.4, 0.0, 1],  
    [18.0, 0.5, 35.0, 0.0, 0.0, 0.5, 2],  
    [7.0, 0.6, 26.0, 0.5, 0.0, 0.0, 0],  
    [23.0, 0.7, 39.0, 0.0, 0.3, 0.1, 1],  
    [17.0, 0.5, 34.0, 0.0, 0.0, 0.4, 2],  
    [10.0, 0.4, 25.0, 0.4, 0.0, 0.3, 0],  
    [22.0, 0.6, 37.0, 0.0, 0.4, 0.0, 1],  
    [16.0, 0.4, 33.0, 0.0, 0.0, 0.5, 2],
    [26.0, 0.6, 42.0, 0.0, 0.2, 0.0, 1],
    [11.0, 0.5, 28.0, 0.3, 0.0, 0.2, 0],
    [27.0, 0.5, 43.0, 0.0, 0.3, 0.1, 1],
    [12.0, 0.4, 26.0, 0.4, 0.0, 0.3, 0],
    [28.0, 0.5, 44.0, 0.0, 0.2, 0.0, 1],
    [13.0, 0.4, 25.0, 0.5, 0.0, 0.4, 0],
    [29.0, 0.4, 45.0, 0.0, 0.3, 0.1, 1],
    [14.0, 0.3, 24.0, 0.6, 0.0, 0.5, 0],
    [30.0, 0.4, 46.0, 0.0, 0.2, 0.0, 1],
    [15.0, 0.3, 23.0, 0.7, 0.0, 0.6, 0],
    [31.0, 0.3, 47.0, 0.0, 0.3, 0.1, 1],
    [16.0, 0.2, 22.0, 0.8, 0.0, 0.7, 0],
    [32.0, 0.3, 48.0, 0.0, 0.2, 0.0, 1],
    [17.0, 0.2, 21.0, 0.9, 0.0, 0.8, 0],
    [33.0, 0.2, 49.0, 0.0, 0.3, 0.1, 1],
    [18.0, 0.1, 20.0, 1.0, 0.0, 0.9, 0],
    [34.0, 0.2, 50.0, 0.0, 0.2, 0.0, 1],
    [19.0, 0.1, 19.0, 1.1, 0.0, 1.0, 0],
    [35.0, 0.1, 51.0, 0.0, 0.3, 0.1, 1],
    [20.0, 0.1, 18.0, 1.2, 0.0, 1.1, 0],
    [36.0, 0.1, 52.0, 0.0, 0.2, 0.0, 1],
    [21.0, 0.2, 17.0, 1.3, 0.0, 1.2, 0],
    [37.0, 0.2, 53.0, 0.0, 0.3, 0.1, 1],
    [22.0, 0.2, 16.0, 1.4, 0.0, 1.3, 0],
    [38.0, 0.2, 54.0, 0.0, 0.2, 0.0, 1],
    [23.0, 0.3, 15.0, 1.5, 0.0, 1.4, 0],
    [39.0, 0.3, 55.0, 0.0, 0.3, 0.1, 1],
    [24.0, 0.3, 14.0, 1.6, 0.0, 1.5, 0],
    [40.0, 0.3, 56.0, 0.0, 0.2, 0.0, 1],
    [25.0, 0.4, 13.0, 1.7, 0.0, 1.6, 0],
    [41.0, 0.4, 57.0, 0.0, 0.3, 0.1, 1],
    # Add the new entries here
]

max_location = max([data[0] for data in training_data])
max_fuel = max([data[1] for data in training_data])
max_speed = max([data[2] for data in training_data])
max_acceleration = max([data[3] for data in training_data])
max_braking = max([data[4] for data in training_data])
max_steering = max([data[5] for data in training_data])

# Prepare the training data
X_train = []
y_train = []
for data in training_data:
    state = preprocess_input(data[0], data[1], data[2], data[3], data[4], data[5])
    action = data[6]
    X_train.append(state)
    y_train.append(action)

# Convert training data to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
# Convert target values to one-hot encoded format
y_train_one_hot = to_categorical(y_train, num_classes=3)

# Define the neural network architecture
model = Sequential()
model.add(Dense(32, input_dim=7, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Three actions: accelerate, brake, or maintain speed

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X_train, y_train_one_hot, epochs=100, batch_size=32)

@app.route('/get_suggestion', methods=['POST'])
def get_suggestion():
    data = request.json
    location = data['location']
    fuel = data['fuel']
    speed = data['speed']
    acceleration = data['acceleration']
    braking = data['braking']
    steering = data['steering']
    
    state = preprocess_input(location, fuel, speed, acceleration, braking, steering)
    action_probs = model.predict(np.array([state]))[0]
    suggested_action = np.argmax(action_probs)
    
    if suggested_action == 0:
        return jsonify({"suggested_action": "accelerate"})
    elif suggested_action == 1:
        return jsonify({"suggested_action": "brake"})
    elif suggested_action == 2:
        return jsonify({"suggested_action": "maintain_speed"})

# Simulated cars with their details
cars = [
    Car(12.0, 0.7, 28.0, 0.2, 0.0, 0.0),
    Car(15.0, 0.5, 32.0, 0.0, 0.1, 0.0),
    Car(15.0, 0.5, 32.0, 0.0, 0.1, 1.0), 
    Car(15.0, 0.4, 32.0, 0.0, 0.1, 1.0)
    
    # Add more cars as needed
]

# Update the car details and suggested action
def update_car_details():
    for i, car in enumerate(cars):
        car.location += np.random.uniform(-1, 1)
        car.fuel -= np.random.uniform(0.01, 0.05)
        car.speed += np.random.uniform(-2, 2)
        car.acceleration += np.random.uniform(-0.1, 0.1)
        car.braking += np.random.uniform(-0.1, 0.1)
        car.steering += np.random.uniform(-0.1, 0.1)
        suggested_action = control_traffic(car.location, car.fuel, car.speed, car.acceleration, car.braking, car.steering)
        print("Car Details:", car.get_details())
        print("Suggested Action:", suggested_action)
        print("")

# Start updating the car details and suggested actions
update_car_details()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
