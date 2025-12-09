from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import shutil
from glob import glob
from pymongo import MongoClient
from datetime import datetime, timezone

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
            input_shape = model.input_shape[1:3]  # e.g., (256, 256)
            img = image.load_img(filepath, target_size=input_shape)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # ✅ Use raw image (no preprocess_input) because model includes preprocessing
            pred = model.predict(img_array)
            class_index = np.argmax(pred, axis=1)[0]
            confidence = pred[0][class_index] * 100
            predicted_label = class_labels[class_index]

            prediction = f"{predicted_label} ({confidence:.2f}%)"

            # Store in MongoDB
            try:
                predictions_collection.insert_one({
                    "tag": "image",
                    "filename": filename,
                    "prediction": predicted_label,
                    "confidence": float(f"{confidence:.2f}"),
                    "timestamp": datetime.now(timezone.utc)
                })
                print(f"✅ Stored image prediction for {filename} in MongoDB")
            except Exception as e:
                print(f"❌ Failed to store prediction: {e}")

    return render_template('index.html', prediction=prediction, filename=filename)


# =========================
# Video Prediction Route
# =========================
@app.route('/video', methods=['GET', 'POST'])
def video_predict():
    video_prediction = None
    filename = None
    frames_analyzed = 0
    mean_fake_prob = 0

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No video file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected video file"
        if file:
            # Save uploaded video
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = file.filename

            # Prepare frames folder
            FRAMES_DIR = os.path.join(app.config['UPLOAD_FOLDER'], "frames_temp")
            if os.path.exists(FRAMES_DIR):
                shutil.rmtree(FRAMES_DIR)
            os.makedirs(FRAMES_DIR, exist_ok=True)

            # --- Extract frames at ~2 FPS ---
            cap = cv2.VideoCapture(filepath)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = max(1, int(round(src_fps / 2)))
            frame_idx = 0
            saved = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    frame_path = os.path.join(FRAMES_DIR, f"frame_{saved:05d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved += 1
                frame_idx += 1
            cap.release()

            if saved == 0:
                return "No frames extracted from video"

            print(f"📸 Extracted {saved} frames")
            frames_analyzed = saved

            # --- Predict per frame ---
            frame_files = sorted(glob(os.path.join(FRAMES_DIR, "*.jpg")))
            p_fakes = []

            for f in frame_files:
                img = image.load_img(f, target_size=model.input_shape[1:3])
                arr = image.img_to_array(img)
                arr = np.expand_dims(arr, axis=0)
                # Use raw image (no preprocess_input) because model includes preprocessing
                pred = model.predict(arr, verbose=0)[0]

                # Get fake probability based on model output
                if len(pred) == 1:
                    p_fake = float(pred[0])
                else:
                    # Assuming index 0 is fake, index 1 is real
                    p_fake = float(pred[0])
                p_fakes.append(p_fake)

            # --- Aggregate results ---
            mean_fake = np.mean(p_fakes)
            vote_frac_fake = np.mean([pf > 0.5 for pf in p_fakes])
            is_fake_final = (mean_fake > 0.5) or (vote_frac_fake > 0.5)

            final_label = "fake" if is_fake_final else "real"
            mean_fake_prob = mean_fake * 100

            video_prediction = f"{final_label} ({mean_fake_prob:.2f}%)"

            # Cleanup frames
            shutil.rmtree(FRAMES_DIR)

            print(f"🎬 Video prediction: {video_prediction}")

            # Store in MongoDB
            try:
                predictions_collection.insert_one({
                    "tag": "video",
                    "filename": filename,
                    "prediction": final_label,
                    "confidence": float(f"{mean_fake_prob:.2f}"),
                    "frames_analyzed": frames_analyzed,
                    "timestamp": datetime.now(timezone.utc)
                })
                print(f"✅ Stored video prediction for {filename} in MongoDB")
            except Exception as e:
                print(f"❌ Failed to store video prediction: {e}")

    return render_template('video.html', 
                         prediction=video_prediction, 
                         filename=filename,
                         frames_analyzed=frames_analyzed)


# =========================
# Predictions History Route
# =========================
@app.route('/predictions')
def predictions():
    try:
        # Get all predictions, sorted by timestamp (newest first)
        all_predictions = list(predictions_collection.find().sort("timestamp", -1))
        
        # Convert ObjectId to string for JSON serialization
        for pred in all_predictions:
            pred['_id'] = str(pred['_id'])
            # Format timestamp
            if 'timestamp' in pred:
                pred['timestamp'] = pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')
        
        return render_template('predictions.html', predictions=all_predictions)
    except Exception as e:
        print(f"❌ Failed to fetch predictions: {e}")
        return render_template('predictions.html', predictions=[], error=str(e))


# =========================
# Run App
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
