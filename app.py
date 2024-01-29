from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_svm = load_model("model\model.h5")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


confidence_threshold = 0.8

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image file
        image_file = request.files['image']

        if image_file and allowed_file(image_file.filename):

            # Prepare image for prediction
            img = Image.open(image_file)
            img = img.resize((300, 300))
            x = image.img_to_array(img)
            x = x / 255.0  
            images = np.expand_dims(x, axis=0)

            # Data augmentation
            datagen = ImageDataGenerator(
                rotation_range=40,
                shear_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            # Generate augmented images
            augmented_images = datagen.flow(images)

            # Perform predictions on the augmented images
            prediction_array_svm = model_svm.predict(augmented_images)
            average_prediction = np.mean(prediction_array_svm, axis=0)

            class_names = ['Kepiting Biasa', 'Kepiting Soka']

            # Check confidence level
            confidence_svm = np.max(average_prediction)
            if confidence_svm < confidence_threshold:
                return jsonify({"error": "Kepiting tidak terdeteksi dengan keyakinan yang cukup."}), 400

            # Format the response JSON
            predictions = {
                "prediction_svm": class_names[np.argmax(average_prediction)],
                "confidence_svm": '{:2.0f}%'.format(100 * np.max(average_prediction)),
            }

            return jsonify(predictions)
        else:
            return jsonify({"error": "Invalid file format."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def hello():
    return "Crabify"


if __name__ == '_main_':
    app.run(host='0.0.0.0', debug=True)