import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt

# Custom callback to log the learning rate
class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        print(f"Epoch {epoch+1}: Learning rate is {tf.keras.backend.get_value(lr)}")

# Ensure the data directory exists
data_dir = '/Users/htetaung/Desktop/PJ/flask_app/data/train'
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory '{data_dir}' does not exist. Please check the path.")

# Ensure subdirectories exist
classes = ['COVID', 'Normal', 'Pneumonia']
for cls in classes:
    if not os.path.exists(os.path.join(data_dir, cls)):
        raise FileNotFoundError(f"Subdirectory '{os.path.join(data_dir, cls)}' does not exist. Please check the path.")

# Define the model
model = Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% of the data for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

batch_size = 20  # Define batch size

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=batch_size,  # Set batch size
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training'  # Use the 'training' subset
)
validation_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=batch_size,  # Set batch size
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='validation'  # Use the 'validation' subset
)

# Verify that data generators are not empty
if train_generator.samples == 0:
    raise ValueError(f"No training images found in directory '{data_dir}'. Please check the path and ensure it contains images.")
if validation_generator.samples == 0:
    raise ValueError(f"No validation images found in directory '{data_dir}'. Please check the path and ensure it contains images.")

# Calculate steps per epoch
steps_per_epoch = max(train_generator.samples // train_generator.batch_size, 1)
validation_steps = max(validation_generator.samples // validation_generator.batch_size, 1)

# Callbacks (without EarlyStopping and ReduceLROnPlateau to allow overfitting)
learning_rate_logger = LearningRateLogger()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,  # Increase epochs to ensure overfitting
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[learning_rate_logger]
)

# Save the model
model.save('/Users/htetaung/Desktop/PJ/flask_app/model/covid_detector_model.h5')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save the plot
plt.savefig('/Users/htetaung/Desktop/PJ/flask_app/static/training_history.png')
plt.show()  # Display the plot in the console
