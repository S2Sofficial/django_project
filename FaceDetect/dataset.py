import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load your annotated eye dataset and preprocess it
# You should have a dataset with images and corresponding eye locations (bounding boxes)

# Define your model
model = Sequential()

# Convolutional layer 1
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps
model.add(Flatten())

# Fully connected layers
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='linear')) # Two output units for x and y coordinates of the eye

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Load your dataset and preprocess the images and labels
# Ensure that your dataset is appropriately prepared with images and corresponding eye coordinates

# Train the model
model.fit(images, eye_coordinates, epochs=10, batch_size=32)

# Save the trained model
model.save('eye_detection_model.h5')
