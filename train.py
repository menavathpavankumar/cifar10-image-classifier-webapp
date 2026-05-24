from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Load Dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# -----------------------------
# Normalize Images
# -----------------------------
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# -----------------------------
# One-Hot Encode Labels
# -----------------------------
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -----------------------------
# Data Augmentation
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(x_train)

# -----------------------------
# Build Improved CNN
# -----------------------------
model = Sequential()

# Block 1
model.add(Conv2D(
    32,
    (3, 3),
    padding='same',
    activation='relu',
    input_shape=(32, 32, 3)
))
model.add(BatchNormalization())

model.add(Conv2D(
    32,
    (3, 3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(
    64,
    (3, 3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())

model.add(Conv2D(
    64,
    (3, 3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.30))

# Block 3
model.add(Conv2D(
    128,
    (3, 3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

# Flatten
model.add(Flatten())

# Dense Layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(10, activation='softmax'))

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(x_test, y_test)
)

# -----------------------------
# Save Model
# -----------------------------
model.save("cifar10_model.h5")

print("Improved model saved successfully!")