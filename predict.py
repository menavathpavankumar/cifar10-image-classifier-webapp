import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load trained model
model = load_model('models/cifar10_model.h5')

# Class labels
class_names = ['airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck']

def predict_image(img_path):

    img = image.load_img(img_path, target_size=(32,32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    plt.imshow(img)
    plt.axis('off')
    plt.show()

    print("Predicted Class :", class_names[predicted_class])
    print("Confidence :", round(confidence,2), "%")


predict_image('test_images/test1.png')