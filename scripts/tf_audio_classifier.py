import numpy as np
import json
import sys
import os
import tempfile
import subprocess
import warnings
from scipy.io import wavfile

# Silence all background noise from TensorFlow
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError:
    print("Error: tensorflow or tensorflow-hub is not installed.")
    sys.exit(1)

def map_to_ui_category(subclass_full):
    parts = subclass_full.split("/")
    label = parts[-1] if len(parts) > 1 else subclass_full
    label_lower = label.lower()
    
    mapping = {
        "scream": "Scream / Aggression",
        "gunshot": "Gunshot / Explosion",
        "explosion": "Gunshot / Explosion",
        "siren": "Siren / Alarm",
        "car_alarm": "Siren / Alarm",
        "horn": "Siren / Alarm",
        "footsteps": "Footsteps",
        "glass_shatter": "Impact / Breach",
        "knock": "Impact / Breach",
        "hammer": "Impact / Breach",
        "splash": "Impact / Breach",
        "wind": "Atmospheric Wind",
        "storm": "Atmospheric Wind",
        "rainfall": "Atmospheric Wind",
    }
    
    ui_label = label.replace("_", " ").title()
    
    if label_lower in mapping:
        return mapping[label_lower], ui_label
        
    prefix = parts[0].lower() if len(parts) > 1 else ""
    if prefix == "animals":
        return "Animal Signal", ui_label
    elif prefix == "instrument":
        return "Musical Content", ui_label
    elif prefix == "vehicle":
        return "Vehicle Sound", ui_label
    elif prefix == "human":
        return "Human Voice", ui_label
        
    return "Ambient / Noise", ui_label

def convert_and_normalize(input_path):
    temp_wav = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000', '-ac', '1',
            '-c:a', 'pcm_s16le', temp_wav
        ], check=True, capture_output=True, timeout=60)
        return temp_wav
    except Exception as e:
        print(f"Error in ffmpeg: {e}", file=sys.stderr)
        return None

def classify_audio(audio_path, job_id):
    temp_wav = None
    try:
        audio_path = audio_path.strip('"')
        if not os.path.exists(audio_path):
            return {"status": "error", "message": f"File not found: {audio_path}"}

        print(f"--- Advanced TF Hierarchical Classifier Starting (Job: {job_id}) ---")
        
        temp_wav = convert_and_normalize(audio_path)
        if not temp_wav:
             return {"status": "error", "message": "Failed to normalize audio"}

        print("Model: Loading YAMNet (Stage 1)...")
        # Load from HUB
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

        print("Model: Loading Student Model (Stage 2)...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        custom_dir = os.path.join(script_dir, '..', 'pretrained_models', 'custom')
        model_path = os.path.join(custom_dir, 'student_model_20260315_010813.keras')
        labels_path = os.path.join(custom_dir, 'student_model_20260315_010813_labels.json')
        
        if not os.path.exists(model_path):
            return {"status": "error", "message": "Student model missing in pretrained_models/custom!"}
            
        student_model = tf.keras.models.load_model(model_path, compile=False)
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        sub_classes = labels_data['sub_classes']

        print("Processing: Reading audio data...")
        sample_rate, wav_data = wavfile.read(temp_wav)
        if wav_data.dtype == np.int16:
            wav_data = wav_data.astype(np.float32) / 32768.0

        print("Inference: Running 2-stage analysis...")
        
        # Parameters
        window_size = 16000 * 1  # 1 sec
        step_size = 16000 // 2   # 0.5 sec overlap
        total_samples = len(wav_data)
        
        all_detections = []

        for i in range(0, total_samples, step_size):
            window = wav_data[i : i + window_size]
            if len(window) < 16000 // 2: # Ignore too small chunk at the end
                break
                
            timestamp = i / 16000.0
            
            # Stage 1: YAMNet
            scores, embeddings, spec = yamnet_model(window)
            
            if embeddings.shape[0] == 0:
                continue
                
            emb_mean = tf.reduce_mean(embeddings, axis=0)
            emb_max = tf.reduce_max(embeddings, axis=0)
            final_emb = tf.concat([emb_mean, emb_max], axis=0)
            final_emb = tf.expand_dims(final_emb, 0)
            
            # Stage 2: Student
            main_out, sub_out = student_model(final_emb)
            
            # Extract top sub class
            sub_preds = sub_out.numpy()[0]
            top_idx = np.argmax(sub_preds)
            confidence = float(sub_preds[top_idx])
            
            # Thresholding
            if confidence > 0.4:
                raw_label = sub_classes[top_idx]
                ui_category, ui_label = map_to_ui_category(raw_label)
                
                if ui_category != "Ambient / Noise":
                    decibels = round(20 * np.log10(max(1e-5, confidence)) - 10, 1)
                    
                    print(f"[Student TF] Time: {timestamp:.2f}s | Class: {ui_category} ({ui_label}) | Conf: {confidence:.4f}")
                    
                    # Deduplicate in same window
                    existing = next((d for d in all_detections if d["time"] == round(timestamp, 3) and d["type"] == ui_category), None)
                    if not existing:
                        all_detections.append({
                            "type": ui_category,
                            "label": ui_label,
                            "confidence": confidence,
                            "time": round(timestamp, 3),
                            "decibels": decibels
                        })
                    elif confidence > existing["confidence"]:
                        existing["confidence"] = confidence
                        existing["label"] = ui_label

        required_ui_categories = [
            "Human Voice", "Musical Content", "Gunshot / Explosion", 
            "Siren / Alarm", "Scream / Aggression", "Vehicle Sound",
            "Footsteps", "Animal Signal", "Atmospheric Wind", "Impact / Breach"
        ]
        
        summary_events = []
        for cat in required_ui_categories:
            cat_matches = [d for d in all_detections if d["type"] == cat]
            if cat_matches:
                best = max(cat_matches, key=lambda x: x["confidence"])
                summary_events.append({
                    "category": cat,
                    "status": "DETECTED",
                    "confidence": round(best["confidence"] * 100, 1),
                    "details": f"Confirmed: {best['label']}"
                })
        
        all_detections.sort(key=lambda x: x["time"])

        return {
            "jobID": job_id,
            "duration": round(total_samples / 16000.0, 2),
            "sampleRate": 16000,
            "soundEvents": all_detections,
            "categorySummary": summary_events,
            "allDetections": all_detections,
            "summary": f"Forensic analysis successful. {len(all_detections)} sound events isolated."
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Classification error: {str(e)}"}
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try: os.unlink(temp_wav)
            except: pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: python tf_audio_classifier.py <audio_path> [job_id]"}))
        sys.exit(1)
    
    res = classify_audio(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "job")
    
    if isinstance(res, dict) and res.get("status") == "error":
        print(f"\n[!] Forensic Analysis Failed: {res.get('message')}")
    else:
        num_events = len(res.get("soundEvents", []))
        print(f"\n[+] Forensic Analysis Successful")
        print(f"[+] {num_events} Forensic events identified via Student Model.")

    # Hidden JSON for API parsing - wrapped in markers
    print(f"\n[JSON_START]{json.dumps(res)}[JSON_END]")
