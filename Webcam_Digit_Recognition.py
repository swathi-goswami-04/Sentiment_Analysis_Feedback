import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers, models # type: ignore

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape the data
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Build a simple model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Function to preprocess the image from webcam
def preprocess_image(frame):
    roi = frame[100:300, 100:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    inverted = cv2.bitwise_not(resized)
    normalized = inverted / 255.0
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    return reshaped

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Handwritten Digit Recognition")

while True:
    ret, frame = cap.read()
    processed_image = preprocess_image(frame)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    cv2.putText(frame, f'Predicted Digit: {predicted_digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Handwritten Digit Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()