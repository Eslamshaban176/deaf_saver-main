import numpy as np
from flask import Flask, request, render_template
import librosa
from tensorflow.keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)

# Load the Keras model from the specified file path

loaded_model = load_model("model.keras")

# Mapping of labels to class names
label_to_class = {
    0: 'Knocking Sound', 1: 'Azan', 2: 'Car Horn', 3: 'Cat', 4: 'Church Bell',
    5: 'Clock Alarm', 6: 'Cough', 7: 'Crying Baby', 8: 'Dog Bark', 9: 'Glass Breaking',
    10: 'Gun Shot', 11: 'Rain', 12: 'Siren', 13: 'Train', 14: 'Water Drops', 15: 'Wind'
}

def extract_features_and_predict(file, model):
    # Load audio data with librosa
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')

    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    # Predict using the loaded model
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predictions = model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predictions, axis=1)
    return label_to_class.get(predicted_label[0], "Label not found")

@app.route("/", methods=["GET"])
def home():
    return render_template("chat.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'audio_file' not in request.files:
        return "No file part", 400
    file = request.files['audio_file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Predicting the class of audio file
        prediction = extract_features_and_predict(file, loaded_model)
        return jsonify({"prediction": prediction})
        #return render_template("chat.html", prediction_text=f"Prediction: {prediction}")

# Main function
if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)
