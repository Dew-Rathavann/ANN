import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt # for plotting/visualization
from tensorflow.keras.models import Sequential # for building the ANN
from tensorflow.keras.layers import Dense, Flatten # ANN layers
from tensorflow.keras.datasets import cifar10 # CIFAR-10 demo dataset
from tensorflow.keras.callbacks import Callback # for custom callback during training
from sklearn.metrics import precision_score, recall_score, f1_score # for multi-class metrics calculation of precision, recall, F1 score

(X_train, y_train), (X_test, y_test) = cifar10.load_data() # load CIFAR-10 dataset

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Normalize pixel values from [0, 255] to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Custom callback to calculate Precision, Recall, F1 at the end of each epoch
class MetricsCallback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        # lists to store metrics for plotting
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.accuracies = []
    
    # Calculate metrics at the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0).argmax(axis=1)
        y_true = self.y_val.flatten()
        
        # Calculating Precision, Recall, F1 Score is recommended for multi-class classification
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        accuracy = logs.get('accuracy', 0)
        
        # Store metrics by appending to lists
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1s.append(f1)
        self.accuracies.append(accuracy)
        
        print(f" - precision: {precision:.4f} - recall: {recall:.4f} - f1: {f1:.4f}")

# Build the ANN model
model = Sequential([
    Flatten(input_shape=(32, 32, 3)), # converts the 32×32×3 image (3072 pixels) into a 1D array
    Dense(128, activation='relu'), # hidden layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'), # hidden layer with 64 neurons
    Dense(10, activation='softmax') # output layer with 10 neurons (one per class) and softmax for multi-class
])

model.compile(
    optimizer='adam', # Uses Adam optimizer (adaptive learning rate algorithm)
    loss='sparse_categorical_crossentropy', # for multi-class with integer labels
    metrics=['accuracy'] # Tracks accuracy during training
)

metrics_callback = MetricsCallback(X_test, y_test) # Initialize the metrics callback
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1,
          callbacks=[metrics_callback]) # Train the model with metrics callback

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Evaluation metrics over epochs
epochs_range = range(1, len(metrics_callback.accuracies) + 1)
ax1.plot(epochs_range, metrics_callback.accuracies, 'b-o', label='Accuracy')
ax1.plot(epochs_range, metrics_callback.precisions, 'g-s', label='Precision')
ax1.plot(epochs_range, metrics_callback.recalls, 'r-^', label='Recall')
ax1.plot(epochs_range, metrics_callback.f1s, 'm-d', label='F1 Score')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Score')
ax1.set_title('Training Metrics per Epoch')
ax1.legend()
ax1.grid(True)
ax1.set_xticks(list(epochs_range))
ax1.set_ylim(0, 1)

# Plot 2: Test image prediction
index = 30  # try changing this to test different images
image = X_test[index]
prediction = model.predict(image.reshape(1, 32, 32, 3))[0] # predict on a single test image
predicted_class = prediction.argmax() # get the class with highest probability
confidence = prediction[predicted_class] # get the confidence score
actual_class = y_test[index][0] # actual class label

ax2.imshow(image, interpolation='nearest')
ax2.set_title(f"Prediction: {class_names[predicted_class]} ({confidence:.2f})\nActual: {class_names[actual_class]}", fontsize=14)
ax2.axis("on")

plt.tight_layout()
plt.show()
