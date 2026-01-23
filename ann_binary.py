import tensorflow as tf
import matplotlib.pyplot as plt # for plotting/visualization
from tensorflow.keras.models import Sequential # for building the ANN
from tensorflow.keras.layers import Dense, Flatten # ANN layers Dense = fully connected layer, Flatten = flattening layer (to convert 2D images to 1D vectors)
from tensorflow.keras.datasets import cifar10 # CIFAR-10 demo dataset

(X_train, y_train_orig), (X_test, y_test_orig) = cifar10.load_data() # load CIFAR-10 dataset

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Cat = 1, Not Cat = 0 (class index 3 is cat)
y_train = (y_train_orig == 3).astype(int)
y_test = (y_test_orig == 3).astype(int)

# Normalize pixel values from [0, 255] to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the ANN model
model = Sequential([
    Flatten(input_shape=(32, 32, 3)), # converts the 32×32×3 image (3072 pixels) into a 1D array
    Dense(32, activation='relu'), # hidden layer with 32 neurons and ReLU activation
    Dense(1, activation='sigmoid') # output layer with 1 neuron and sigmoid activation for binary classification
])

model.compile(
    optimizer='adam', # Uses Adam optimizer (adaptive learning rate algorithm)
    loss='binary_crossentropy',
    metrics=['accuracy'] # Tracks accuracy during training
)

model.fit(X_train, y_train, epochs=3, batch_size=128, verbose=1) # Train the model

index = 36  # try changing this
image = X_test[index]
prediction = model.predict(image.reshape(1, 32, 32, 3))[0][0] # predict on a single test image

label = "Cat 🐱" if prediction > 0.5 else "Not Cat ❌"
actual = "Cat 🐱" if y_test[index] == 1 else "Not Cat ❌"
original_class = class_names[y_test_orig[index][0]]

plt.figure(figsize=(6, 6))
plt.imshow(image, interpolation='nearest')
plt.title(f"Prediction: {label} ({prediction:.2f})\nActual: {actual} (It's a {original_class})", fontsize=14)
plt.axis("on")
plt.tight_layout()
plt.show()
