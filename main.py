from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Print dataset shapes
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Show first image
plt.imshow(x_train[0])
plt.title(f"Class Label: {y_train[0][0]}")
plt.show()