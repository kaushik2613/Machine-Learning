import numpy as np
import os as os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler



class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.d_input = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, input_data):
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, d_output):
        raise NotImplementedError("Backward pass not implemented.")


class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(self.output_size, self.input_size)
        self.biases = np.random.randn(self.output_size)

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights.T) + self.biases
        return self.output

    def backward(self, d_output):
        self.d_weights = np.dot(d_output.T, self.input)
        self.d_biases = np.sum(d_output, axis=0)
        self.d_input = np.dot(d_output, self.weights)
        return self.d_input

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

    def get_weights(self):
        return self.weights, self.biases

    def set_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases


class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + np.exp(-input_data))  # Sigmoid activation
        return self.output

    def backward(self, d_output):
        sigmoid_derivative = self.output * (1 - self.output)
        self.d_input = d_output * sigmoid_derivative
        return self.d_input

class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        """
        Forward pass for the ReLU activation function.
        input_data: shape (n, d)
        """
        self.input = input_data
        self.output = np.maximum(0, input_data)  # ReLU activation
        return self.output

    def backward(self, d_output):
        """
        Backward pass for the ReLU activation function.
        d_output: Gradient of the loss w.r.t the output of this layer, shape (n, h)
        """
        # Derivative of ReLU: 1 for positive inputs, 0 for non-positive inputs
        self.d_input = d_output * (self.input > 0)  # Element-wise multiplication
        return self.d_input



class BinaryCrossEntropyLoss(Layer):
    def __init__(self):
        self.predicted = None
        self.true_labels = None

    def forward(self, predicted, true_labels):
        self.predicted = predicted
        self.true_labels = true_labels
        self.predicted = np.clip(self.predicted, 1e-9, 1 - 1e-9)

        # Compute binary cross-entropy loss
        loss = -np.mean(self.true_labels * np.log(self.predicted) + (1 - self.true_labels) * np.log(1 - self.predicted))
        return loss

    def backward(self):
        gradient = (self.predicted - self.true_labels) / (self.predicted * (1 - self.predicted))
        return gradient


class Sequential(Layer):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, d_output):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)
        return d_output

    def update_weights(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(learning_rate)

    def save_weights(self, file_path):
        """
        Save the weights and biases of all layers to a file.
        """
        weights_dict = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):  # Only save weights and biases for LinearLayer
                weights, biases = layer.get_weights()
                weights_dict[f'layer_{idx}'] = {'weights': weights, 'biases': biases}

        # Save weights_dict to a .npy file
        np.save(file_path, weights_dict)
        print(f"Weights saved to {file_path}")

    def load_weights(self, file_path):
        """
        Load the weights and biases from a file into the layers.
        """
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return

        weights_dict = np.load(file_path, allow_pickle=True).item()

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):  # Only load weights and biases for LinearLayer
                if f'layer_{idx}' in weights_dict:
                    weights_dict_layer = weights_dict[f'layer_{idx}']
                    weights = weights_dict_layer['weights']
                    biases = weights_dict_layer['biases']
                    layer.set_weights(weights, biases)

        print(f"Weights loaded from {file_path}")

# XOR Dataset
samples = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(samples[: ,0], samples[: ,1], c = targets)
plt.show()


# Define Tanh Activation Function
class Tanh(Layer):
    def forward(self, inputs):
        # Store the input for backward pass
        self.input = inputs
        # Apply the Tanh function
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, gradient):
        # Derivative of Tanh is (1 - tanh(x)^2)
        tanh_derivative = 1 - self.output ** 2
        # Return the gradient scaled by the derivative of the Tanh function
        return gradient * tanh_derivative


# Sigmoid Activation
sigmoid_model = Sequential()
sigmoid_model.add(LinearLayer(2, 2))
sigmoid_model.add(SigmoidLayer())
sigmoid_model.add(LinearLayer(2, 1))
sigmoid_model.add(SigmoidLayer())

# Tanh Activation
tanh_model = Sequential()
tanh_model.add(LinearLayer(2, 2))
tanh_model.add(Tanh())
tanh_model.add(LinearLayer(2, 1))
tanh_model.add(SigmoidLayer())

# Define Loss Function
loss_function = BinaryCrossEntropyLoss()

epochs = 10000
learning_rate = 0.1

# Training Function
def test_model_XOR(model, name):
    for epoch in range(epochs):
        outputs = model.forward(samples)
        loss = loss_function.forward(outputs, targets)
        grad = loss_function.backward()
        model.backward(grad)
        model.update_weights(learning_rate)

        # Print loss after every 10 epochs

        if epoch % 100 == 0:
            print(f"{name} - Epoch {epoch}: Loss {loss}")

    # Final predictions
    output = model.forward(samples)
    print(f"Final predictions for {name}:")
    print(output)
    print(f"Final rounded predictions for {name}:")
    print(np.round(output))

    # Save weights
    model.save_weights(f"{name}_XOR_solved.w")

# Train both models
test_model_XOR(sigmoid_model, "Sigmoid")
test_model_XOR(tanh_model, "Tanh")


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Loading the dataset
dataset = np.load("nyc_taxi_data.npy", allow_pickle=True).item()
X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]

print(X_train.head())
print(X_train.tail())

print(y_train.head())
print(y_train.tail())


# Print only column names
print(X_train.columns)
## Summary of the dataset
X_train.info()

## descriptive summary of the dataset
X_train.describe()


# Calculate the sum of missing values for each column
missing_values_sum = X_train.isna().sum()

# Print the result
print("Sum of missing values for each column:")
print(missing_values_sum)

#Drop missing values
df = X_train.dropna()

## Duplicate records
print("Duplicate values:")
print(X_train[X_train.duplicated()])


# Identify numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
numerical_columns_test = X_test.select_dtypes(include=['float64', 'int64']).columns

selected_features_plot = [ 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
df_plot = df[selected_features_plot]

# Select only numerical columns
numerical_df = df_plot.select_dtypes(include=['number'])

# Create subplots - one for each numerical column
n_columns = len(numerical_df.columns)

# Set up the figure and axes
fig, axes = plt.subplots(1, n_columns, figsize=(15, 6))

# Plot box plot for each numerical column
for i, column in enumerate(numerical_df.columns):
    sns.boxplot(x=numerical_df[column], ax=axes[i])
    axes[i].set_title(f'Box Plot of {column}')
    axes[i].set_xlabel('Values')

# Display the plot
plt.tight_layout()
plt.show()
# Plot histograms of each column in the same figure
n_columns = len(df_plot.columns)  # Number of columns
n_rows = (n_columns // 2) + (n_columns % 2)  # Number of rows for subplots

plt.figure(figsize=(12, 6))  # Set the figure size

for i, column in enumerate(df_plot.columns):
    plt.subplot(n_rows, 2, i + 1)  # Create a subplot for each column
    plt.hist(df_plot[column], bins=10, edgecolor='black')  # Plot histogram
    plt.title(f'Histogram of {column}')  # Set title for each subplot
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=y_train)
plt.title('Boxplot of Trip Duration')
plt.show()


y_train.hist(bins=10, edgecolor='black')  # You can adjust the number of bins
plt.title('Histogram of trip_duration')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Create a MinMaxScaler object
scaler2 = StandardScaler()

# Apply Min-Max scaling only to numerical columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
X_test[numerical_columns_test] = scaler.transform(X_test[numerical_columns_test])

print("Scaled DataFrame:")
print(df)


df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
df['pickup_month'] = df['pickup_datetime'].dt.month
df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['dropoff_month'] = df['dropoff_datetime'].dt.month
df['dropoff_day'] = df['dropoff_datetime'].dt.day
df['dropoff_hour'] = df['dropoff_datetime'].dt.hour
X_test['pickup_datetime'] = pd.to_datetime(X_test['pickup_datetime'])
X_test['dropoff_datetime'] = pd.to_datetime(X_test['dropoff_datetime'])
X_test['pickup_month'] = X_test['pickup_datetime'].dt.month
X_test['pickup_day'] = X_test['pickup_datetime'].dt.day
X_test['pickup_hour'] = X_test['pickup_datetime'].dt.hour
X_test['dropoff_month'] = X_test['dropoff_datetime'].dt.month
X_test['dropoff_day'] = X_test['dropoff_datetime'].dt.day
X_test['dropoff_hour'] = X_test['dropoff_datetime'].dt.hour

selected_features = ['dropoff_month', 'dropoff_day', 'dropoff_hour', 'pickup_month', 'pickup_day', 'pickup_hour', 'passenger_count', 'id', 'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']
df = df[selected_features]
X_test = X_test[selected_features]
print(df)
df.info()


# Identify categorical columns (object or category type)
categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns
print("Categorical Columns:", categorical_columns)

# Convert all object columns to integers (if possible)
for col in categorical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

for col in categorical_columns:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(int)

print(df.head())
df.info()

print(df.dtypes)
print(X_test.dtypes)

# Combine df1 and df2 horizontally (as columns)
df_combined = pd.concat([df, y_train], axis=1)

print(df_combined)

#Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df_combined.corr(),annot=True)
plt.show()

## Correlation
print(df.corr())

# Calculate the correlation of all columns in df1 with 'trip_duration'
correlations = df_combined.corr()['trip_duration'].drop('trip_duration')

# Sort the correlations in descending order and get the top 5 features
top_5_features = correlations.abs().sort_values(ascending=False).head(5)

# Print the top 5 features
print("Top 5 features correlated with trip_duration:")
print(top_5_features)


# selected_features_corr = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'vendor_id']
selected_features_corr = ['dropoff_month', 'dropoff_day', 'dropoff_hour', 'pickup_month', 'pickup_day', 'pickup_hour', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
# Select only those features from df_combined
selected_features_df = df_combined[selected_features_corr]
X_test1 = X_test[selected_features_corr]
# Print the selected features
print("Selected features:")
print(selected_features_df)
print(X_test1.info())
selected_features_df.info()



# Converting to NumPy arrays
X_numpy = selected_features_df.to_numpy(dtype=np.float32)
y_numpy = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
X_test_numpy = X_test1.to_numpy(dtype=np.float32)
y_test_numpy = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)

#Normalisization
X_numpy = (X_numpy - X_numpy.mean()) / X_numpy.std()
X_test_numpy = (X_test_numpy - X_test_numpy.mean()) / X_test_numpy.std()


# Spliting into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_numpy, y_numpy, test_size=0.2, random_state=42)
X_val = (X_val - X_val.mean()) / X_val.std()


class RMSLE:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon  # Small constant to avoid log(0) and division by zero

    def forward(self, predictions, targets):
        """Compute RMSLE loss."""
        predictions = np.clip(predictions, self.epsilon, None)  # Clip values to avoid log(0)
        targets = np.clip(targets, self.epsilon, None)

        # Ensure both arrays are the same length
        min_length = min(len(predictions), len(targets))
        predictions = predictions[:min_length]
        targets = targets[:min_length]

        log_diff = np.log1p(predictions) - np.log1p(targets)
        squared_log_diff = np.square(log_diff)

        return np.sqrt(np.mean(squared_log_diff))

    def backward(self, predictions, targets):
        """Compute gradient for backpropagation."""
        predictions = np.clip(predictions, self.epsilon, None)
        targets = np.clip(targets, self.epsilon, None)

        # Ensure both arrays are the same length
        min_length = min(len(predictions), len(targets))
        predictions = predictions[:min_length]
        targets = targets[:min_length]

        n_samples = len(targets)
        log_diff = np.log1p(predictions) - np.log1p(targets)

        # Direct gradient computation (no need for self.forward)
        grad = (2 * log_diff) / (n_samples * (predictions + 1))

        return grad

# Define Neural Network Architectures

model1 = Sequential()
model1.add(LinearLayer(X_train.shape[1], 64))
model1.add(ReLULayer())
model1.add(LinearLayer(64, 32))
model1.add(ReLULayer())
model1.add(LinearLayer(32, 1))

model2 = Sequential()
model2.add(LinearLayer(X_train.shape[1], 8))
model2.add(ReLULayer())
model2.add(LinearLayer(8, 8))
model2.add(ReLULayer())
model2.add(LinearLayer(8, 4))
model2.add(ReLULayer())
model2.add(LinearLayer(4, 1))

model3 = Sequential()
model3.add(LinearLayer(X_train.shape[1], 8))
model3.add(ReLULayer())
model3.add(LinearLayer(8, 8))
model3.add(ReLULayer())
model3.add(LinearLayer(8, 4))
model3.add(ReLULayer())
model3.add(LinearLayer(4, 4))
model3.add(ReLULayer())
model3.add(LinearLayer(4, 1))



def train_validate_test(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=100, initial_learning_rate=0.01,
                        batch_size=32, patience=3, lr_decay=0.95, min_improvement=1e-4):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_count = 0  # Counter for early stopping
    loss_fn = RMSLE()  # Your loss function
    learning_rate = initial_learning_rate

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        train_loss_epoch = 0
        val_loss_epoch = 0

        # Train in batches
        for i in range(0, len(X_train_shuffled), batch_size):
            batch_X = X_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]

            train_output = model.forward(batch_X)
            train_loss = loss_fn.forward(train_output, batch_y)
            train_gradient = loss_fn.backward(train_output, batch_y)  # Call backward without arguments

            model.backward(train_gradient)  # Ensure model's backward computes gradients
            model.update_weights(learning_rate)  # Update model weights

            train_loss_epoch += train_loss * len(batch_X)

        # Validate the model
        val_outputs = model.forward(X_val)
        val_loss_epoch = loss_fn.forward(val_outputs, y_val)

        # Average losses over batches
        train_loss_epoch /= len(X_train_shuffled)
        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)

        # Early stopping and learning rate decay logic
        print(
            f"Epoch {epoch + 1}, Validation Loss: {val_loss_epoch:.4f}, Best Validation Loss: {best_val_loss:.4f}, Patience Count: {patience_count}")

        # Check if the validation loss improved by more than the min_improvement threshold
        if val_loss_epoch < best_val_loss - min_improvement:  # If validation loss improves significantly
            print("Validation loss improved.")
            best_val_loss = val_loss_epoch
            patience_count = 0  # Reset patience count since loss improved
        else:  # If validation loss doesn't improve significantly
            patience_count += 1
            print(f"Validation loss did not improve, patience count: {patience_count}")

            if patience_count >= patience:  # If patience exceeds, stop early
                print(f"Early stopping at epoch {epoch + 1}.")
                break  # Exit training loop

        # Apply learning rate decay only if patience hasn't been exhausted
        if patience_count < patience:
            learning_rate *= lr_decay

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss_epoch:.4f}, Validation Loss: {val_loss_epoch:.4f}, LR: {learning_rate:.6f}")

    # Calculate and report test loss
    test_outputs = model.forward(X_test)
    test_loss = loss_fn.forward(test_outputs, y_test)
    print(f"Test Loss: {test_loss:.4f}")

    return train_losses, val_losses, test_loss


# Train, Validate, and Test the Models
train_losses_1, val_losses_1, test_losses_1 = train_validate_test(model1, X_train, y_train, X_val, y_val, X_test_numpy, y_test_numpy)
train_losses_2, val_losses_2, test_losses_2 = train_validate_test(model2, X_train, y_train, X_val, y_val, X_test_numpy, y_test_numpy)
train_losses_3, val_losses_3, test_losses_3 = train_validate_test(model3, X_train, y_train, X_val, y_val, X_test_numpy, y_test_numpy)

import matplotlib.pyplot as plt

# Plot for Model 1
plt.figure(figsize=(10, 6))
plt.plot(train_losses_1, label='Training Loss', color='blue')
plt.plot(val_losses_1, label='Validation Loss', color='lightblue', linestyle='--')
plt.title('Model 1 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSLE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for Model 2
plt.figure(figsize=(10, 6))
plt.plot(train_losses_2, label='Training Loss', color='green')
plt.plot(val_losses_2, label='Validation Loss', color='lightgreen', linestyle='--')
plt.title('Model 2 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSLE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for Model 3
plt.figure(figsize=(10, 6))
plt.plot(train_losses_3, label='Training Loss', color='red')
plt.plot(val_losses_3, label='Validation Loss', color='salmon', linestyle='--')
plt.title('Model 3 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSLE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output test loss for each model
print(f"Model 1 Test Loss: {test_losses_1:.4f}")
print(f"Model 2 Test Loss: {test_losses_2:.4f}")
print(f"Model 3 Test Loss: {test_losses_3:.4f}")