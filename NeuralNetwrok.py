import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense
# Define the neural network architecture
input_dim = 7 + 9
model = Sequential()
model.add(Dense(32, activation='relu', name='input_layer'))
model.add(Dense(32, activation='relu', name='hidden_layer1'))
model.add(Dense(3, activation='softmax', name='output_layer'))

# Create a plotly figure
fig = go.Figure()

# Add nodes for each layer
for layer in model.layers:
    if isinstance(layer, Dense):
        # Add nodes for each neuron in the layer
        for i in range(layer.units):
            fig.add_trace(go.Scatter(x=[layer.name], y=[i], mode='markers', marker=dict(size=10)))

# Connect nodes with edges
for i in range(len(model.layers) - 1):
    layer1 = model.layers[i]
    layer2 = model.layers[i + 1]

    if isinstance(layer1, Dense) and isinstance(layer2, Dense):
        for i in range(layer1.units):
            for j in range(layer2.units):
                fig.add_trace(go.Scatter(x=[layer1.name, layer2.name], y=[i, j], mode='lines'))

# Customize layout
fig.update_layout(title_text="Neural Network Architecture", showlegend=False)

# Show the figure
fig.show()