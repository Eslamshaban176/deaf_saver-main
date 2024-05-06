import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import librosa
import tensorflow as tf
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__)

label_to_class = {
    0: 'Knocking_Sound', 1: 'azan', 2: 'car_horn', 3: 'cat', 4: 'church bell', 
    5: 'clock_alarm', 6: 'cough', 7: 'crying_baby', 8: 'dog_bark', 9: 'glass_breaking', 
    10: 'gun_shot', 11: 'rain', 12: 'siren', 13: 'train', 14: 'water_drops', 15: 'wind'
}

def get_class_name(label):
    return label_to_class.get(label, "Label not found")

def extract_features_and_predict(filename, loaded_model):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')

        # Extract MFCC features
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        # Make predictions
        predicted_probabilities = loaded_model.predict(mfccs_scaled_features)
        predicted_label = np.argmax(predicted_probabilities, axis=1)
        prediction_class = [get_class_name(label) for label in predicted_label]

        return prediction_class
    except Exception as e:
        return str(e)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', 'wav'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask routes
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/predict_route", methods=["POST"])
def predict_route():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio_file']
    
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if audio_file and allowed_file(audio_file.filename):
        filename = secure_filename(audio_file.filename)
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(audio_file_path)

        prediction = extract_features_and_predict(audio_file_path, loaded_model)

        os.remove(audio_file_path)  # Remove the uploaded file after processing

        if isinstance(prediction, list):
            return jsonify({"prediction": prediction})
        else:
            return jsonify({"error": prediction}), 500
    else:
        return jsonify({"error": "Invalid file format. Supported formats: mp3, wav"}), 400
    
# Main function
if __name__ == "__main__":
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
