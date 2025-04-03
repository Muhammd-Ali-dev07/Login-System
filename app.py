from flask import Flask, request, jsonify
import mysql.connector
import librosa
import numpy as np
import os
import io
import cv2
import face_recognition
import tempfile
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ Connect to MySQL Database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="ali.07070707.pk",
    database="neurogate"
)
cursor = db.cursor()

# ✅ Step 1: Test Connection
@app.route("/")
def home():
    return "<h1>✅ MySQL Connection Test Successful!</h1>"

# ✅ Step 1: Fetch All Users
@app.route("/get_users", methods=["GET"])
def get_users():
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    user_list = []
    for user in users:
        user_list.append({
            "id": user[0],
            "name": user[1],
            "gender": user[2],
            "unique_id": user[3]
        })
    return jsonify(user_list)

# ✅ Step 1: Register New User
@app.route("/add_user", methods=["POST"])
def add_user():
    data = request.get_json()
    name = data.get("name")
    gender = data.get("gender")
    unique_id = data.get("unique_id")
    try:
        cursor.execute("INSERT INTO users (name, gender, unique_id) VALUES (%s, %s, %s)", (name, gender, unique_id))
        db.commit()
        return jsonify({"message": "✅ User added successfully!"})
    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 409

# ✅ Step 2: Upload Voice Sample + Extract MFCC
@app.route("/upload_voice", methods=["POST"])
def upload_voice():
    import tempfile
    from pydub import AudioSegment  # pip install pydub

    unique_id = request.form["unique_id"]
    audio_file = request.files["audio"]
    audio_data = audio_file.read()

    try:
        # ✅ Save original audio to DB
        cursor.execute("UPDATE users SET voice = %s WHERE unique_id = %s", (audio_data, unique_id))
        db.commit()

        # ✅ Save audio to a temporary WebM file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
            temp_webm.write(audio_data)
            webm_path = temp_webm.name

        # ✅ Convert WebM to proper WAV using pydub and ffmpeg
        wav_path = webm_path.replace(".webm", ".wav")
        try:
            sound = AudioSegment.from_file(webm_path)
            sound.export(wav_path, format="wav")

            # ✅ Load WAV with librosa
            import librosa
            y, sr = librosa.load(wav_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mean_vector = np.mean(mfcc.T, axis=0)

            # ✅ Clean up
            os.remove(webm_path)
            os.remove(wav_path)

            return jsonify({
                "message": "✅ Voice saved successfully!",
                "mfcc_matrix": mfcc.tolist(),
                "mean_vector": mean_vector.tolist()
            })

        except Exception as e:
            return jsonify({"error": f"Conversion error (pydub/ffmpeg): {str(e)}"}), 500

    except Exception as err:
        return jsonify({"error": f"Server error: {str(err)}"}), 500

# ✅ Step 2: Verify Voice Sample
@app.route("/verify_voice", methods=["POST"])
def verify_voice():
    unique_id = request.form["unique_id"]
    new_audio = request.files["audio"]
    new_audio_data = new_audio.read()

    cursor.execute("SELECT voice FROM users WHERE unique_id = %s", (unique_id,))
    result = cursor.fetchone()

    if result and result[0]:
        stored_audio, _ = librosa.load(io.BytesIO(result[0]), sr=16000)
        new_audio, _ = librosa.load(io.BytesIO(new_audio_data), sr=16000)
        stored_features = extract_features(stored_audio)
        new_features = extract_features(new_audio)

        similarity = np.dot(stored_features, new_features) / (
            np.linalg.norm(stored_features) * np.linalg.norm(new_features))

        if similarity > 0.85:
            return jsonify({"message": "✅ Voice Match! Proceed to Step 3"}), 200
        else:
            return jsonify({"error": "❌ Voice Mismatch! Try again."}), 401

    return jsonify({"error": "⚠️ No registered voice found!"}), 404

# ✅ Step 3: Capture Face Dataset
@app.route("/capture_faces", methods=["POST"])
def capture_faces():
    unique_id = request.form["unique_id"]
    folder_path = os.path.join("faces", unique_id)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "❌ Camera not accessible."}), 500

    count = 0
    while count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)

        if faces:
            count += 1
            filename = os.path.join(folder_path, f"{unique_id}_{count}.jpg")
            cv2.imwrite(filename, frame)

    cap.release()
    cv2.destroyAllWindows()

    try:
        cursor.execute("UPDATE users SET face_folder = %s WHERE unique_id = %s", (folder_path, unique_id))
        db.commit()
        return jsonify({"message": "✅ Face images captured and stored."})
    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

# ✅ Feature extraction helper
def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

if __name__ == "__main__":
    app.run(debug=True)
