from flask import Flask, request, jsonify
import tensorflow.lite as tflite
import numpy as np
import os

app = Flask(__name__)
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'input' not in data:
        return jsonify({"error": "No input data provided"}), 400
    input_data = np.array(data['input'], dtype=np.float32).reshape(1, 28, 28, 1)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    prediction = np.argmax(output)
    probabilities = output[0].tolist()
    return jsonify({"prediction": int(prediction), "probabilities": probabilities})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
