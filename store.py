from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from pymongo import MongoClient
from datetime import datetime

# =========================
# Flask App Config
# =========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# =========================
# Load trained model
# =========================
MODEL_PATH = 'best_model_v3.keras'
model = load_model(MODEL_PATH)
print(f"✅ Loaded model from {MODEL_PATH}")

# =========================
# Label mapping
# =========================
class_labels = {0: "fake", 1: "real"}

# =========================
# MongoDB Setup
# =========================
# Make sure authSource=admin because root user is created in admin DB
MONGO_URI = "mongodb://admin:secret123@localhost:27017/predictions_db?authSource=admin"
client = MongoClient(MONGO_URI)
db = client['predictions_db']
predictions_collection = db['predictions']

# =========================
# Routes
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save uploaded file
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = file.filename

            # Load and preprocess image
            input_shape = model.input_shape[1:3]
            img = image.load_img(filepath, target_size=input_shape)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            pred = model.predict(img_array)
            class_index = np.argmax(pred, axis=1)[0]
            confidence = pred[0][class_index] * 100
            predicted_label = class_labels[class_index]
            prediction = f"{predicted_label} ({confidence:.2f}%)"

            # Store in MongoDB
            try:
                predictions_collection.insert_one({
                    "filename": filename,
                    "prediction": predicted_label,
                    "confidence": float(f"{confidence:.2f}"),
                    "timestamp": datetime.utcnow()
                })
                print(f"✅ Stored prediction for {filename} in MongoDB")
            except Exception as e:
                print(f"❌ Failed to store prediction: {e}")

    return render_template('index.html', prediction=prediction, filename=filename)

# =========================
# Run App
# =========================
if __name__ == '__main__':
    app.run(debug=True)
