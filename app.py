from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import json

app = Flask(__name__)
model = None


def load_my_model():
    global model
    model_path = 'phase3_with_dropout_l2.keras'
    model = load_model(model_path)
    model.make_predict_function()


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        load_my_model()

    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.0

    result = model.predict(np.expand_dims(image, axis=0))

    with open('class_mapping.json', 'r') as f:
        index_to_class = json.load(f)

    predicted_class = np.argmax(result)
    predicted_class_name = index_to_class[str(predicted_class)]

    return jsonify({'attraction': predicted_class_name})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
