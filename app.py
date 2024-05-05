import numpy as np
from flask import Flask, request, render_template, jsonify
import librosa
import tensorflow as tf

from tensorflow.keras.models import load_model

# Load the model from the .keras file
loaded_model = load_model('model.keras')

# Now you can use the loaded_model for inference or further processing
# Load the saved model and labelencoder
from tensorflow.keras.models import load_model # type: ignore

# Flask app initialization
flask_app = Flask(__name__)

label_to_class = {
    0: 'Knocking_Sound', 1: 'azan', 2: 'car_horn', 3: 'cat', 4: 'church bell', 
    5: 'clock_alarm', 6: 'cough', 7: 'crying_baby', 8: 'dog_bark', 9: 'glass_breaking', 
    10: 'gun_shot', 11: 'rain', 12: 'siren', 13: 'train', 14: 'water_drops', 15: 'wind'
}


def get_class_name(label):
    if label in label_to_class:
        return label_to_class[label]
    else:
        return "Label not found"
    
    
def extract_features_and_predict(filename, loaded_model):
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

# Load the saved model
loaded_model = load_model("model.keras")

# Load the saved model
# loaded_model = tf.saved_model.load('saved_model')

def make_prediction(filename):
    prediction = extract_features_and_predict(filename, loaded_model)
    return prediction

# Flask routes
@flask_app.route("/")
def home():
    return render_template("chat.html")

@flask_app.route("/predict_route", methods=["POST"])
def predict_route():
    if 'audio_file' not in request.files:
        return "No audio file uploaded", 400
    audio_file = request.files['audio_file']
    if not audio_file:
        audio_file = "audio.mp3"

    prediction = make_prediction(audio_file)

    # return jsonify({"prediction": prediction})
    return render_template("chat.html", prediction_text=f"{prediction}".title())

# Main function
if __name__ == "__main__":
    flask_app.run(debug=True)
    