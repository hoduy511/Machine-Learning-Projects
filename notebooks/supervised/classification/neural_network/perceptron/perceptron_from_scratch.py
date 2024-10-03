import numpy as np


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize the perceptron with random weights and set the learning rate
        self.weights = np.random.rand(input_size + 1)  # +1 for the bias term
        self.learning_rate = learning_rate

    def activation_function(self, x):
        # Activation function that returns 1 if x is greater than 0, otherwise returns 0
        return 1 if x > 0 else 0

    def predict(self, inputs):
        # Calculate the weighted sum of inputs and apply the activation function
        weighted_sum = np.dot(
            self.weights, np.insert(inputs, 0, 1)
        )  # Insert bias term (1) at the beginning
        return self.activation_function(weighted_sum)

    def train(self, X_train, y_train, epochs):
        # Train the perceptron for a specified number of epochs
        for epoch in range(epochs):
            converged = True  # Track convergence status
            for i in range(len(X_train)):
                # Make a prediction for the current input
                prediction = self.predict(X_train[i])
                # If prediction is incorrect, update the weights
                if prediction != y_train[i]:
                    error = y_train[i] - prediction  # Calculate the error
                    # Update weights based on the error and learning rate
                    self.weights += (
                        self.learning_rate * error * np.insert(X_train[i], 0, 1)
                    )  # Include bias in update
                    converged = False  # Mark as not converged
            # If no weight updates were made, the model has converged
            if converged:
                print(f"Converged at epoch {epoch + 1} and weight {self.weights}")
                break
