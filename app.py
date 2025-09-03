from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
import os

# ----------------------------
# Load model and tokenizer once at startup
MODEL_PATH = "cyberbullying_detection_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 500

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

# ----------------------------
# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.form["comment"]

    # Convert text to sequence
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=MAX_SEQUENCE_LENGTH)

    # Predict
    prediction = model.predict(input_padded)[0][0]
    result = "Cyberbullying" if prediction > 0.5 else "Non-Cyberbullying"

    return render_template("index.html", text=input_text, result=result)

# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
