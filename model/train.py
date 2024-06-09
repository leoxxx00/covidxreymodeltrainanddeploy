import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Ensure the data directory exists
data_dir = '/Users/htetaung/Desktop/PJ/flask_app/data/train'
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory '{data_dir}' does not exist. Please check the path.")

# Ensure subdirectories exist
classes = ['COVID', 'Normal', 'Pneumonia']
for cls in classes:
    if not os.path.exists(os.path.join(data_dir, cls)):
        raise FileNotFoundError(f"Subdirectory '{os.path.join(data_dir, cls)}' does not exist. Please check the path.")

# Create train and validation directories
train_dir = '/Users/htetaung/Desktop/PJ/flask_app/data/train_split'
val_dir = '/Users/htetaung/Desktop/PJ/flask_app/data/val_split'

for dir_path in [train_dir, val_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for cls in classes:
        cls_dir = os.path.join(dir_path, cls)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)


# Split the dataset using numpy
def train_test_split(files, test_size=0.2, shuffle=True, random_state=None):
    if shuffle:
        if random_state:
            np.random.seed(random_state)
        np.random.shuffle(files)
    split_idx = int(len(files) * (1 - test_size))
    return files[:split_idx], files[split_idx:]


# Split the dataset and copy files
for cls in classes:
    cls_dir = os.path.join(data_dir, cls)
    images = os.listdir(cls_dir)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    for img in train_images:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(train_dir, cls, img)
        shutil.copyfile(src, dst)

    for img in val_images:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(val_dir, cls, img)
        shutil.copyfile(src, dst)

# Define the model
model = Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)
validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

# Verify that data generators are not empty
if train_generator.samples == 0:
    raise ValueError(
        f"No training images found in directory '{train_dir}'. Please check the path and ensure it contains images.")
if validation_generator.samples == 0:
    raise ValueError(
        f"No validation images found in directory '{val_dir}'. Please check the path and ensure it contains images.")

# Calculate steps per epoch
steps_per_epoch = max(train_generator.samples // train_generator.batch_size, 1)
validation_steps = max(validation_generator.samples // validation_generator.batch_size, 1)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=3,  # Increased epochs for better training
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('/Users/htetaung/Desktop/PJ/flask_app/model/covid_detector_model.h5')

# Plot training & validation accuracy values
plt.figure(figsize=(18, 10))

# Accuracy plot
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Loss plot
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Learning Rate plot
plt.subplot(2, 2, 3)
plt.plot(history.history['lr'], label='Learning Rate')
plt.title('Learning Rate')
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Validation steps plot
plt.subplot(2, 2, 4)
plt.plot(range(len(history.history['val_loss'])), [validation_steps] * len(history.history['val_loss']),
         label='Validation Steps')
plt.title('Validation Steps per Epoch')
plt.ylabel('Validation Steps')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Save the plot
plt.tight_layout()
plt.savefig('/Users/htetaung/Desktop/PJ/flask_app/static/training_history.png')
plt.show()  # Display the plot in the console
#datalink-https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9937995/#:~:text=The%20COVID%2D19%20subjects%2C%20who,and%20bilateral%20predominance%20%5B3%5D.
