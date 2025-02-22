import numpy as np
from NN import NeuralNetworks
import matplotlib.pyplot as plt


nn = NeuralNetworks(layer_sizes=[2, 10, 1], activation_fun=["relu", "sigmoid"])

# Generattion of the training data:
# I will  sample points uniformly from a square covering [-1.5, 1.5] in both x and y.
n_samples = 3000
X = np.random.uniform(-1.5, 1.5, (n_samples, 2))
# Label: 1 if the point is inside the unit circle, else 0.
y = (np.sum(X**2, axis=1) <= 1.0).astype(float)

# Training parameters
epochs = 1000
learning_rate = 0.1
losses = []

# Training loop: Process one sample at a time.
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(n_samples):
        # Reshape sample to be (features, 1)
        x_sample = X[i].reshape(2, 1)
        # Reshape label to be (1, 1)
        y_sample = np.array([[y[i]]])
        nn.backpropagation(x_sample, y_sample, learning_rate)
        sample_loss = nn.calculate_loss(y_sample)
        epoch_loss += sample_loss
    avg_loss = epoch_loss / n_samples
    losses.append(avg_loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {avg_loss}")



# Create a grid of points over the square [-1.5, 1.5] x [-1.5, 1.5]
grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

# Evaluate the network on each grid point
preds = np.zeros(len(grid_points))
for i in range(len(grid_points)):
    inp = grid_points[i].reshape(2, 1)
    preds[i] = nn.forward_pass(inp)
preds = preds.reshape(grid_x.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 8))
# Contour: regions where output is near 0.5 will approximate the decision boundary.
plt.contourf(grid_x, grid_y, preds, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'blue'])
plt.colorbar(label='NN output')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.6)
plt.title("Circle Classification Using Neural Network")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title("Training Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
