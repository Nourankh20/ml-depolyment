import requests
import numpy as np
import tensorflow as tf

# Load a sample MNIST image
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
sample_image = x_test[0].reshape(1, 28, 28, 1).astype('float32') / 255.0

# Send request to the API
url = 'http://localhost:5000/predict'
data = {'input': sample_image.flatten().tolist()}
response = requests.post(url, json=data)

# Print response
print(response.json())