from NN import NeuralNetworks
import numpy as np
import matplotlib.pyplot as plt


nn = NeuralNetworks(layer_sizes=[1, 10, 1], activation_fun=["relu", "linear"])

# Generate training data from a linear function: y = 2x + 1
x = np.linspace(-10, 10, 100).reshape(-1, 1)
y = 2 * x + 1

# Normalize the data to the range [-1, 1] 
def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    norm_data = 2 * (data - data_min) / (data_max - data_min) - 1
    return norm_data, data_min, data_max

def denormalize(norm_data, data_min, data_max):
    return (norm_data + 1) / 2 * (data_max - data_min) + data_min

x_norm, x_min, x_max = normalize(x)
y_norm, y_min, y_max = normalize(y)

epochs = 1000
learning_rate = 0.001  #Using a lower learning rate for stability
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(len(x_norm)):
        x_sample = x_norm[i:i+1]  # shape (1,1)
        y_sample = y_norm[i:i+1]  # shape (1,1)
        nn.backpropagation(x_sample, y_sample, learning_rate)
        sample_loss = nn.calculate_loss(y_sample)
        epoch_loss += sample_loss
    avg_loss = epoch_loss / len(x_norm)
    losses.append(avg_loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {avg_loss}")

# Get predictions for all normalized x values.
predictions_norm = np.zeros_like(x_norm)
for i in range(len(x_norm)):
    x_sample = x_norm[i:i+1]
    predictions_norm[i:i+1] = nn.forward_pass(x_sample)

# Denormalize predictions back to original scale.
predictions = denormalize(predictions_norm, y_min, y_max)

# Plot the original data and the predictions.
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='True Data')
plt.plot(x, predictions, color='red', label='Predictions')
plt.title("Linear Regression with Normalized Input and Output")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Plot the loss history
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Loss over epochs')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History")
plt.legend()
plt.show()
