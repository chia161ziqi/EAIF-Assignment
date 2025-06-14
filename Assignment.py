import pandas as pd
import numpy as np
import os # Reading files from directories
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Read image data from dataset
def load_images(filename):
    with open(filename, 'rb') as f: # Opens the file in binary read mode ('rb')
        magic = int.from_bytes(f.read(4), 'big') # Magic number to identify the file type and convert these bytes into an integer using big-endian byte order
        if magic != 2051: # Magic number should be 2051 (standard for MNIST images)
            raise ValueError(f'Invalid magic number {magic} in image file: {filename}') # The file is probably not a valid MNIST image file if not
        num_images = int.from_bytes(f.read(4), 'big') # Reads the next 4 bytes, representing the number of images in the file
        num_rows = int.from_bytes(f.read(4), 'big') # Reads the next 4 bytes, representing the number of rows per image
        num_cols = int.from_bytes(f.read(4), 'big') # Reads the next 4 bytes, representing the number of columns per image
        print(f'Loading {num_images} images of size {num_rows}x{num_cols}') # Prints information about the data being loaded.
        image_data = f.read(num_images * num_rows * num_cols) # Reads the raw image pixel data for all images, each pixel is 1 byte
        images = np.frombuffer(image_data, dtype=np.uint8) # Converts the raw bytes into a 1D NumPy array of type uint8 (values from 0 to 255).
        images = images.reshape((num_images, num_rows, num_cols)) # Reshapes the flat array into a 3D array of shape
        return images # Returns the NumPy array containing all the images

# Read label from dataset
def load_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2049: # For label files, the expected magic number is 2049
            raise ValueError(f'Invalid magic number {magic} in label file: {filename}')
        num_labels = int.from_bytes(f.read(4), 'big') # Reads the next 4 bytes to get the total number of labels in the file and converts the bytes to an integer
        print(f'Loading {num_labels} labels')
        label_data = f.read(num_labels) # Reads the label data from the file, each label is 1 byte
        labels = np.frombuffer(label_data, dtype=np.uint8) # Converts the raw byte data into a NumPy array of type uint8
        return labels # Returns the NumPy array containing all the labels

# Modify this to point to our MNIST .ubyte files folder
data_dir = "C:/Users/Asus/PycharmProjects/eaif/"

# Load training data
x_train = load_images(os.path.join(data_dir, "train-images.idx3-ubyte"))
y_train = load_labels(os.path.join(data_dir, "train-labels.idx1-ubyte"))

# Load test data
x_test = load_images(os.path.join(data_dir, "t10k-images.idx3-ubyte"))
y_test = load_labels(os.path.join(data_dir, "t10k-labels.idx1-ubyte"))

# Print the shape
print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

# Normalize image data
x_train = x_train.astype('float32') / 255.0 # Divides all pixel values by 255 to scale them from 0–255 → 0–1
x_test = x_test.astype('float32') / 255.0

# Reshape the data
x_train = x_train.reshape(-1, 28, 28) # Reshapes the training data to have shape (number of images, 28, 28).
x_test = x_test.reshape(-1, 28, 28)

# Display dataset
class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
    plt.xlabel(class_names[y_train[i]])
plt.show()

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)), # Converts each 28x28 image into a flat 1D array of 784 values (28×28=784).

    Dense(512), # Each neuron will learn to recognize different features from the input
    BatchNormalization(), # Normalizes the outputs of the previous layer
    LeakyReLU(alpha=0.1), # Allows a small gradient (alpha=0.1) when the input is negative to avoid “dead neurons.”
    Dropout(0.2), # Randomly turns off 20% of the neurons during training to prevent overfitting

    Dense(256),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),

    Dense(10, activation='softmax') # Final dense layer with 10 neurons (one for each digit 0–9)
                                    # Uses softmax activation to output probabilities that sum to 1, predicting the digit class.
])

# When the model fails to improve for three consecutive steps on the validation data, terminate the training early and retain the best version of the model.
early_stop = EarlyStopping(
    monitor='val_loss', # Observe val_loss
    patience=3, # doesn’t improve for 3 epochs in a row, stop training
    restore_best_weights=True # Store the best weights
)

# Compile the model
model.compile(optimizer=Adam(), # Uses the Adam optimizer to help the model learn and update weights efficiently.
              loss=SparseCategoricalCrossentropy(), # This is the loss function used for classification when your labels are integers (like 0–9).
              metrics=[SparseCategoricalAccuracy()]) # Tracks how accurate the model is during training and testing.

# Train the model
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[early_stop])
# model.fit(...): Starts the training process.
# x_train, y_train: Training images and labels.
# epochs=50: Train the model for up to 50 rounds.
# validation_data=(x_test, y_test): Uses test data to check how well the model is doing after each epoch.
# callbacks=[early_stop]: Uses the early stopping rule to stop training early if validation loss stops improving.

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) # runs the model on the test images and labels, returning the loss and accuracy.
print(f"Test accuracy: {test_acc * 100:.2f}%")

result = history.history
print(result)

# Plot Training and Validation Accuracy Over Epochs
plt.plot(result['sparse_categorical_accuracy'], label='accuracy')
plt.plot(result['val_sparse_categorical_accuracy'], label = 'val_accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1.5])
plt.legend(loc='lower right')
plt.show()

# Plot Training and Validation Loss Over Epochs
plt.plot(result['loss'], label='loss')
plt.plot(result['val_loss'], label = 'val_loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 0.15])
plt.legend(loc='lower right')
plt.show()

# Evaluate the performance of a trained classification model on test data
y_test_pred = model.predict(x_test)
print(y_test_pred.shape)

y_test_pred_classes = np.argmax(y_test_pred,axis = 1)
print(y_test_pred_classes.shape)

print("Classification Report:")
print(classification_report(y_test, y_test_pred_classes))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_classes))

# Predict for inference image
inference_image = x_test[1000]
print(inference_image.max(),inference_image.min())

plt.imshow(inference_image, cmap='gray')  # Use 'cmap' for grayscale images
plt.title("Inference Image")
plt.axis('off')  # Optional: hides axes
plt.show()

print(np.argmax(model.predict(inference_image.reshape(1,28,28))))