from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import tensorflow as tf
import gdown

# ----------------------------
# Download model at runtime
file_id = '1pRTclhvvpUNzExEr2a70b5gR4TCPuYUz'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'cyberbullying_detection_model.h5'

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load the model
model = tf.keras.models.load_model(output)

# Load tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

max_sequence_length = 500

# ----------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['comment']
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)
    prediction = model.predict(input_padded)[0][0]
    result = "Cyberbullying" if prediction > 0.5 else "Non-Cyberbullying"
    return render_template('index.html', text=input_text, result=result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
