import os
import librosa
import numpy as np
import joblib
import glob
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Update emotion mappings to match frontend expectations
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", 
    "04": "sad", "05": "angry", "06": "fear", 
    "07": "disgust", "08": "surprise"
}

EMODB_EMOTIONS = {
    "w": "angry", "l": "boredom", "e": "disgust",
    "a": "fear", "f": "happy", "t": "sad", 
    "n": "neutral"
}

TESS_EMOTIONS = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "pleasant_surprise": "surprise"
}

# Feature extraction function
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y = librosa.effects.trim(y)[0]  # Remove silence
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Feature error: {str(e)}")
        return None

# Function to load and process datasets
def load_dataset(zip_path, unzip_path, dataset_name):
    if not os.path.exists(zip_path):
        print(f"Error: Couldn't find dataset at {zip_path}")
        return [], []
    
    print(f"Unzipping dataset: {zip_path}")
    try:
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        print(f"Dataset unzipped to: {unzip_path}")
        # Print contents of unzip_path to debug
        print(f"Contents of {unzip_path}: {os.listdir(unzip_path)}")
    except Exception as e:
        print(f"Failed to unzip {zip_path}: {e}")
        return [], []

    data = []
    labels = []
    # Update to recursively search subdirectories
    dataset_path = os.path.join(unzip_path, "**", "*.wav")
    print(f"Scanning for audio files in {unzip_path} with pattern {dataset_path}...")
    
    wav_files = glob.glob(dataset_path, recursive=True)
    if not wav_files:
        print(f"No audio files found in {unzip_path}")
        return [], []
    
    if dataset_name == "RAVDESS":
        emotions = RAVDESS_EMOTIONS
        def get_emotion(file_name):
            parts = file_name.split("-")
            return emotions.get(parts[2]) if len(parts) > 2 else None
    elif dataset_name == "EMODB":
        emotions = EMODB_EMOTIONS
        def get_emotion(file_name):
            return emotions.get(file_name[5].lower()) if len(file_name) > 5 else None
    elif dataset_name == "TESS":
        emotions = TESS_EMOTIONS
        def get_emotion(file_name):
            for key in emotions:
                if key in file_name:
                    return emotions[key]
            return None
    else:
        print(f"Unrecognized dataset in {unzip_path}. Skipping...")
        return [], []
    
    total_files = len(wav_files)
    for i, file in enumerate(wav_files, 1):
        file_name = os.path.basename(file).lower()
        emotion_label = get_emotion(file_name)
        
        if emotion_label:
            features = extract_features(file)
            if features is not None:
                data.append(features)
                labels.append(emotion_label)
                print(f"[{i}/{total_files}] Processed {file_name} -> Emotion: {emotion_label}")
        else:
            print(f"[{i}/{total_files}] Skipped {file_name}: No valid emotion label")
    
    return data, labels

# Training function with updated paths and accuracy
def train_model():
    datasets = [
        {"zip_path": r"C:\Users\asim2\OneDrive\Desktop\miniproject\RAVDESS.zip", "unzip_path": r"C:\Users\asim2\OneDrive\Desktop\miniproject\RAVDESS", "name": "RAVDESS"},
        {"zip_path": r"C:\Users\asim2\OneDrive\Desktop\miniproject\EMODB.zip", "unzip_path": r"C:\Users\asim2\OneDrive\Desktop\miniproject\EMODB", "name": "EMODB"},
        {"zip_path": r"C:\Users\asim2\OneDrive\Desktop\miniproject\TESS.zip", "unzip_path": r"C:\Users\asim2\OneDrive\Desktop\miniproject\TESS", "name": "TESS"}
    ]
    
    all_data = []
    all_labels = []
    print("Starting dataset loading...")
    for dataset in datasets:
        data, labels = load_dataset(dataset["zip_path"], dataset["unzip_path"], dataset["name"])
        if data and labels:
            all_data.extend(data)
            all_labels.extend(labels)
            print(f"Loaded {len(data)} samples from {dataset['unzip_path']}")
        else:
            print(f"No samples loaded from {dataset['unzip_path']}")

    if not all_data or not all_labels:
        print("Error: No valid audio data to train on. Check your datasets!")
        return False

    X = np.array(all_data)
    y = np.array(all_labels)
    print(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as 'scaler.pkl'")
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("Training emotion detection model...")
    svm = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    
    # Calculate and print training accuracy
    y_train_pred = svm.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    joblib.dump(svm, "svm_model.pkl")
    print("Model saved as 'svm_model.pkl'")
    print("Training completed successfully!")
    return True

if __name__ == "__main__":
    success = train_model()
    if not success:
        print("Training process ended with errors.")
    else:
        print("Training process completed successfully!")
