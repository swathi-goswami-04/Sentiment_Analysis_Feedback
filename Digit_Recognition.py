import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess the data
# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data to add channel dimension for convolutional layers (28x28x1)
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))


# Step 3: Display images for each digit (0 to 9) in training and testing sets
def display_images_by_digit(images, labels, dataset_name='Dataset'):
    """
    Display images grouped by their digit label (0-9)
    """
    num_digits = 10  # Digits 0-9
    plt.figure(figsize=(10, 10))

    for i in range(num_digits):
        # Get the indices of the images that correspond to digit i
        digit_indices = np.where(labels == i)[0]

        # Display the first 5 images of the current digit
        for j in range(5):
            idx = digit_indices[j]
            plt.subplot(num_digits, 5, i * 5 + j + 1)  # Position the images in a grid
            plt.imshow(images[idx].reshape(28, 28), cmap='gray')
            plt.title(f"Label: {i}")
            plt.axis('off')

    plt.suptitle(f'{dataset_name} Images (Grouped by Digit)', fontsize=16)
    plt.tight_layout()
    plt.show()


# Display 5 random training images for each digit (0 to 9)
display_images_by_digit(x_train, y_train, dataset_name='Training')

# Display 5 random testing images for each digit (0 to 9)
display_images_by_digit(x_test, y_test, dataset_name='Testing')

# Step 4: Build the neural network model
model = models.Sequential([
    # Convolutional Layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3
                       ), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten the output to feed into a fully connected layer
    layers.Flatten(),
    layers.Dense(64, activation='relu'),

    # Output layer with 10 neurons (one for each digit 0-9)
    layers.Dense(10, activation='softmax')
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
model.fit(x_train, y_train, epochs=5)

# Step 7: Evaluate the model

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Step 8: Make predictions
predictions = model.predict(x_test)

# Visualize the predictions for the first few images in the test set
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
    plt.show()