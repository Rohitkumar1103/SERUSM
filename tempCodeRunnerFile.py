import os
import time
import numpy as np
import librosa
import joblib
import subprocess
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pydub import AudioSegment
from pydub.utils import get_encoder_name
import logging
import traceback

# Configure FFmpeg path (UPDATE THIS TO YOUR FFMPEG PATH)
FFMPEG_PATH = r"C:\Users\KIIT\Downloads\Voice Emotion Analysis\ffmpeg-7.1.1\ffmpeg.exe"

# Verify FFmpeg
try:
    output = subprocess.run([FFMPEG_PATH, "-version"], capture_output=True, text=True)
    if output.returncode != 0:
        raise FileNotFoundError(f"FFmpeg not working properly: {output.stderr}")
    print(f"FFmpeg Version: {output.stdout.splitlines()[0]}")
except Exception as e:
    raise RuntimeError(f"FFmpeg check failed: {str(e)}")

# Set FFmpeg path for pydub
AudioSegment.converter = FFMPEG_PATH

app = Flask(__name__)
CORS(app)

# Configuration with absolute paths
UPLOAD_FOLDER = os.path.abspath('uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o777)  # Set full permissions

app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,
    'ALLOWED_EXTENSIONS': {'wav', 'mp3', 'ogg', 'flac', 'webm'},
    'MODEL_PATH': 'svm_model.pkl',
    'SCALER_PATH': 'scaler.pkl'
})

logging.basicConfig(level=logging.DEBUG)

# Load components
try:
    model = joblib.load(app.config['MODEL_PATH'])
    scaler = joblib.load(app.config['SCALER_PATH'])
except Exception as e:
    logging.error(f"Failed to load model/scaler: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_wav(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file missing: {file_path}")

        print(f"Converting: {file_path}")  # Debugging log
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(22050).set_channels(1)
        
        wav_path = os.path.splitext(file_path)[0] + "_converted.wav"
        audio.export(wav_path, format="wav", codec="pcm_s16le")

        print(f"Converted WAV file saved at: {wav_path}")  # Debugging log
        return wav_path
    except Exception as e:
        logging.error(f"Conversion failed: {traceback.format_exc()}")
        raise RuntimeError(f"Audio conversion error: {str(e)}")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        logging.error(f"Feature extraction failed: {traceback.format_exc()}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format'}), 400

    save_path = process_path = None
    try:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        print(f"File saved at: {save_path}")  # Debugging log
        
        if not os.path.exists(save_path):
            raise FileNotFoundError("Temporary file not created")

        process_path = save_path
        if not filename.lower().endswith('.wav'):
            process_path = convert_to_wav(save_path)

        print(f"Processed file at: {process_path}")  # Debugging log

        features = extract_features(process_path)
        if features is None:
            raise ValueError("Feature extraction failed")

        features_scaled = scaler.transform([features])
        emotion = model.predict(features_scaled)[0]
        return jsonify({'emotion': emotion})

    except Exception as e:
        logging.error(f"Processing error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

    finally:
        time.sleep(0.5)  # Allow file handles to release
        for path in [save_path, process_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logging.warning(f"Cleanup failed for {path}: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
