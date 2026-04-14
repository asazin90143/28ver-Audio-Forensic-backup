import sys
import os
import json
import subprocess
import shutil
import warnings
import numpy as np
from scipy.io import wavfile
import tempfile
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]


# FORCE SILENCE
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def convert_to_wav_if_needed(input_path, log_func):
    """
    Tries to read the file. If it fails, converts to WAV using FFmpeg.
    Returns (path_to_read, is_temp)
    """
    try:
        # Check if readable
        try:
            wavfile.read(input_path)
            return input_path, False
        except Exception:
            log_func(f"Direct read failed, attempting conversion for {input_path}")
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp.close()
            output_path = tmp.name
            
            cmd = ['ffmpeg', '-y', '-i', input_path, '-ar', '44100', '-ac', '2', output_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            log_func(f"Converted to {output_path}")
            return output_path, True
    except Exception as e:
        log_func(f"Conversion failed: {str(e)}")
        return input_path, False

def separate_audio(input_path, output_dir, job_id, classification_path=None):
    debug_log = []
    
    def log(msg):
        msg_str = str(msg)
        debug_log.append(msg_str)
        # Print to stderr to pass through the API's error logger (keeps terminal clean)
        # but with a professional prefix
        print(f"[Demucs] {msg_str}", file=sys.stderr)

    converted_audio_path = None
    is_temp_file = False

    try:
        log(f"Start separation. Input: {input_path}, Job: {job_id}")
        input_path = os.path.abspath(input_path.strip('"'))
        output_dir = os.path.abspath(output_dir.strip('"'))
        
        # 0. Ensure Input is Valid WAV
        read_path, is_temp = convert_to_wav_if_needed(input_path, log)
        converted_audio_path = read_path
        is_temp_file = is_temp
        
        if classification_path:
            classification_path = os.path.abspath(classification_path.strip('"'))
        
        # 1. Run Demucs
        use_fast = os.environ.get("USE_FAST_SEPARATION", "false").lower() == "true"
        
        if use_fast:
            log("FAST MODE ENABLED: Using distilled Student Separator (198K params)...")
            from speechbrain.inference.separation import SepformerSeparation
            import torch.nn as nn
            
            class StudentSeparator(nn.Module):
                def __init__(self, input_dim=256, hidden_dim=128, num_sources=3):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Conv1d(input_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm1d(hidden_dim)
                    )
                    self.separator = nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, 1), nn.ReLU(),
                        nn.Conv1d(hidden_dim, hidden_dim, 1), nn.ReLU(),
                        nn.Conv1d(hidden_dim, hidden_dim, 1), nn.ReLU()
                    )
                    self.decoder = nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, 1), nn.ReLU(),
                        nn.Conv1d(hidden_dim, input_dim * num_sources, 1)
                    )
                    self.num_sources = num_sources
                    self.input_dim = input_dim

                def forward(self, x):
                    encoded = self.encoder(x)
                    separated = self.separator(encoded)
                    decoded = self.decoder(separated)
                    b, _, t = decoded.size()
                    return decoded.view(b, self.num_sources, self.input_dim, t)
            
            # Load Teacher's Encoder/Decoder
            teacher = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj03mix",
                savedir=os.path.join(tempfile.gettempdir(), 'speechbrain_models', 'sepformer-wsj03mix')
            )
            
            # Load custom Student Masknet
            student = StudentSeparator()
            student_path = r"C:\Users\picha\OneDrive\Desktop\CODES\training\models\separation\student_separator\student_separator.pt"
            if os.path.exists(student_path):
                student.load_state_dict(torch.load(student_path, map_location="cpu", weights_only=False))
                log("✅ Custom Student Separator weights loaded successfully.")
            else:
                log("⚠️ Student separator weights not found, using raw initialized params.")
            
            student.eval()
            
            # Process via student pipeline
            sr, audio_data = wavfile.read(read_path)
            if audio_data.dtype != np.float32:
                 audio_data = audio_data.astype(np.float32) / 32768.0
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1) # Mono required for SpeechBrain
            
            import torchaudio.transforms as T
            if sr != 8000:
                resampler = T.Resample(sr, 8000)
                wav = resampler(torch.tensor(audio_data).unsqueeze(0))
            else:
                wav = torch.tensor(audio_data).unsqueeze(0)
            
            log("Running lightweight student inference...")
            with torch.no_grad():
                encoded_feat = teacher.mods.encoder(wav)
                masks = student(encoded_feat)
                sources = teacher.mods.decoder(masks)
            
            if sr != 8000:
                upsampler = T.Resample(8000, sr)
                sources_up = upsampler(sources)
            else:
                sources_up = sources
                
            sources_np = sources_up.squeeze(0).numpy() # Shape [3, Time]
            stem_names = ["voice_1", "voice_2", "voice_3"]
        else:
            log("Loading model htdemucs...")
            
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            import torchaudio.transforms as T
            
            model = get_model("htdemucs")
            model.cpu()
            model.eval()
            
            sr, audio_data = wavfile.read(read_path)
            
            if audio_data.dtype == np.int16:
                 audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                 audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                 audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                 
            if len(audio_data.shape) == 1:
                audio_data = np.expand_dims(audio_data, axis=0)
            else:
                audio_data = audio_data.T
                
            if audio_data.shape[0] == 1:
                audio_data = np.concatenate([audio_data, audio_data], axis=0)
                
            wav = torch.tensor(audio_data)
            
            if sr != model.samplerate:
                log(f"Resampling {sr} -> {model.samplerate}Hz")
                resampler = T.Resample(sr, model.samplerate)
                wav = resampler(wav)
                
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()
            
            import multiprocessing
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            log(f"Separating using {num_workers} workers...")
            
            sources = apply_model(model, wav[None], device="cpu", shifts=1, split=True, 
                                 overlap=0.25, progress=True, num_workers=num_workers)[0]
            
            sources = sources * ref.std() + ref.mean()
            sources_np = sources.numpy()
            stem_names = model.sources

        
        log("Separation finished. Saving stems...")
        
        demucs_folder_name = os.path.splitext(os.path.basename(read_path))[0]
        separated_folder = os.path.join(output_dir, "htdemucs", demucs_folder_name)
        os.makedirs(separated_folder, exist_ok=True)
        
        final_stems = {}
        for i, name in enumerate(stem_names):
            out_path = os.path.join(separated_folder, f"{name}.wav")
            # Stereo save if HTDemucs, mono if Student
            if len(sources_np.shape) > 2:
                 data_to_save = sources_np[i].T
                 sample_rate_to_save = model.samplerate
            else:
                 data_to_save = np.expand_dims(sources_np[i], axis=1) # Fake stereo
                 data_to_save = np.concatenate([data_to_save, data_to_save], axis=1)
                 sample_rate_to_save = sr # Used original resampled SR
            
            wavfile.write(out_path, sample_rate_to_save, (data_to_save * 32767).astype(np.int16))
            final_stems[name] = f"/separated_audio/htdemucs/{demucs_folder_name}/{name}.wav"

        # 2. Harmonic/Percussive Masking (Forensic Polish)
        log("Starting forensic masking...")
        try:
            import librosa
            if len(sources_np.shape) > 2 and "htdemucs" in demucs_folder_name:
                merged_background = (sources_np[0] + sources_np[1] + sources_np[2]) / 3.0
                y_back = merged_background[0]
            else:
                y_back = sources_np[0]  # Just use voice 1 as back for Student path
            
            # Harmonic/Percussive Split for Forensic Clarity
            h, p = librosa.effects.hpss(y_back, margin=(1.0, 5.0))
            
            back_path = os.path.join(separated_folder, "background_mixed.wav")
            wavfile.write(back_path, model.samplerate, (p * 32767).astype(np.int16))
            final_stems["background"] = f"/separated_audio/htdemucs/{demucs_folder_name}/background_mixed.wav"
            log("Background masking complete.")
        except Exception as e:
            log(f"Masking Exception: {str(e)}")

        # 3. Gated Stem Extraction (Advanced Forensic)
        if classification_path and os.path.exists(classification_path):
            try:
                with open(classification_path, 'r') as f:
                    cls_data = json.load(f)
                
                import re
                job_id_clean = re.sub(r'[^a-zA-Z0-9]', '_', job_id).lower()
                gen_dir = os.path.join(output_dir, "generated", job_id_clean)
                os.makedirs(gen_dir, exist_ok=True)
                
                log(f"Loaded classification data. Keys: {list(cls_data.keys())}")
                
                # Use the Master Mix for Forensic Gating to guarantee Demucs does not erase valid high-frequency sounds (dogs/sirens).
                y_full, sr_full = librosa.load(read_path, sr=None)
                forensic_base_audio = y_full
                forensic_sr = sr_full
                
                log(f"Using Raw Master Mix for Forensic Gating (Fallback). SR: {forensic_sr}, Shape: {forensic_base_audio.shape}")
                
                # Enhanced Forensic Separation with Distance-Based Grouping
                forensic_targets = {
                    "gunshots": ["Gunshot / Explosion"],
                    "screams": ["Scream / Aggression"],
                    "sirens": ["Siren / Alarm"],
                    "impact": ["Impact / Breach"],
                    "footsteps": ["Footsteps"],
                    "animals": ["Animal Signal"],
                    "wind": ["Atmospheric Wind"],
                    "vehicles": ["Vehicle Sound"],
                    "human_voice": ["Human Voice"],
                    "music": ["Musical Content"]
                }
                
                all_events = cls_data.get("allDetections", [])
                log(f"Processing {len(all_events)} detected forensic events with distance-based separation...")
                
                # Group events by distance ranges for better separation
                def get_distance_range(decibels):
                    db = float(decibels)
                    if db > -20: return "very_close"  # 0-20m
                    elif db > -40: return "close"      # 20-40m  
                    elif db > -60: return "medium"     # 40-60m
                    else: return "far"                  # 60m+
                
                for key, types in forensic_targets.items():
                    matches = [e for e in all_events if e["type"] in types]
                    
                    if matches:
                        log(f"Found {len(matches)} events for {key}")
                        
                        # Create distance-based stems
                        distance_groups = {}
                        for match in matches:
                            dist_range = get_distance_range(match.get("decibels", -60))
                            if dist_range not in distance_groups:
                                distance_groups[dist_range] = []
                            distance_groups[dist_range].append(match)
                        
                        # Generate separate audio for each distance range
                        for dist_range, events in distance_groups.items():
                            out_y = np.zeros_like(forensic_base_audio)
                            
                            for event in events:
                                start_s = event["time"]
                                # Extend window based on distance (farther sounds have longer reverb/tail)
                                window_size = 0.5 if dist_range in ["very_close", "close"] else 1.0
                                end_s = start_s + window_size
                                
                                start_idx = int(start_s * forensic_sr)
                                end_idx = int(end_s * forensic_sr)
                                
                                if end_idx < len(forensic_base_audio):
                                    # Apply fade in/out for smoother transitions
                                    fade_samples = int(0.05 * forensic_sr)  # 50ms fade
                                    segment = np.copy(forensic_base_audio[start_idx:end_idx])
                                    
                                    # Apply fade in
                                    if len(segment) > fade_samples * 2:
                                        fade_in = np.linspace(0, 1, fade_samples)
                                        segment[:fade_samples] *= fade_in
                                        
                                        # Apply fade out
                                        fade_out = np.linspace(1, 0, fade_samples)
                                        segment[-fade_samples:] *= fade_out
                                    
                                    out_y[start_idx:end_idx] = segment
                            
                            # Save distance-grouped stem
                            f_path = os.path.join(gen_dir, f"{key}_{dist_range}.wav")
                            wavfile.write(f_path, forensic_sr, (out_y * 32767).astype(np.int16))
                            final_stems[f"{key}_{dist_range}"] = f"/separated_audio/generated/{job_id_clean}/{key}_{dist_range}.wav"
                            log(f"Created {key}_{dist_range} with {len(events)} events")
                        
                        # Also create a combined stem for this category
                        combined_out_y = np.zeros_like(forensic_base_audio)
                        for match in matches:
                            start_s = match["time"]
                            end_s = start_s + 0.8  # Slightly longer window for combined
                            start_idx = int(start_s * forensic_sr)
                            end_idx = int(end_s * forensic_sr)
                            if end_idx < len(forensic_base_audio):
                                combined_out_y[start_idx:end_idx] = forensic_base_audio[start_idx:end_idx]
                        
                        combined_path = os.path.join(gen_dir, f"{key}.wav")
                        wavfile.write(combined_path, forensic_sr, (combined_out_y * 32767).astype(np.int16))
                        final_stems[key] = f"/separated_audio/generated/{job_id_clean}/{key}.wav"
                        log(f"Created combined {key} stem")
                        
                        # --- Multi-Speaker Isolation (Phase 5 Diarization) ---
                        if key == "human_voice":
                            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
                            if hf_token:
                                log(f"HUGGINGFACE_TOKEN detected. Starting multi-speaker diarization for N-speakers...")
                                try:
                                    from pyannote.audio import Pipeline
                                    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
                                    import torch
                                    if torch.cuda.is_available():
                                        pipeline.to(torch.device("cuda"))
                                        
                                    diarization = pipeline(read_path)
                                    
                                    # Group segments by speaker
                                    speaker_segments = {}
                                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                                        if speaker not in speaker_segments:
                                            speaker_segments[speaker] = []
                                        speaker_segments[speaker].append((turn.start, turn.end))
                                        
                                    log(f"Diarization complete! Found {len(speaker_segments)} unique speakers.")
                                    
                                    # Create isolated stems per speaker
                                    isolated_voices_list = []
                                    for speaker, segs in speaker_segments.items():
                                        spk_y = np.zeros_like(forensic_base_audio)
                                        for start_s, end_s in segs:
                                            # Apply padding for natural isolation
                                            start_s = max(0, start_s - 0.2)
                                            end_s = min(len(forensic_base_audio) / forensic_sr, end_s + 0.2)
                                            
                                            start_idx = int(start_s * forensic_sr)
                                            end_idx = int(end_s * forensic_sr)
                                            
                                            if end_idx <= len(forensic_base_audio):
                                                segment = np.copy(forensic_base_audio[start_idx:end_idx])
                                                
                                                # Smooth fade
                                                fade_len = int(0.05 * forensic_sr)
                                                if len(segment) > fade_len * 2:
                                                    segment[:fade_len] *= np.linspace(0, 1, fade_len)
                                                    segment[-fade_len:] *= np.linspace(1, 0, fade_len)
                                                    
                                                spk_y[start_idx:end_idx] = segment
                                        
                                        spk_path = os.path.join(gen_dir, f"human_voice_{speaker.lower()}.wav")
                                        wavfile.write(spk_path, forensic_sr, (spk_y * 32767).astype(np.int16))
                                        
                                        url = f"/separated_audio/generated/{job_id_clean}/human_voice_{speaker.lower()}.wav"
                                        final_stems[f"human_voice_{speaker.lower()}"] = url
                                        isolated_voices_list.append({"name": f"{speaker.replace('SPEAKER_', 'Speaker ')}", "url": url})
                                        log(f"Isolated {speaker} successfully.")
                                        
                                    final_stems["isolatedVoices"] = isolated_voices_list
                                
                                except Exception as d_err:
                                    log(f"Diarization skipped/failed: {d_err}")
                        
                    else:
                        # Don't mark as empty, create a silent stem instead
                        silent_path = os.path.join(gen_dir, f"{key}_silent.wav")
                        wavfile.write(silent_path, forensic_sr, np.zeros(len(forensic_base_audio), dtype=np.int16))
                        final_stems[key] = f"/separated_audio/generated/{job_id_clean}/{key}_silent.wav"
                        log(f"Created silent stem for {key}")
                
                log("Reconstruction complete.")
            except Exception as e:
                log(f"Forensic Gating Exception: {str(e)}")

        if not final_stems:
             log("No stems were generated.")
             return {"status": "error", "message": "Separation failed, no stems found.", "debug": debug_log}

        return {"status": "success", "stems": final_stems, "debug": debug_log}
    except Exception as e:
        return {"status": "error", "message": str(e), "debug": debug_log}
    finally:
        if is_temp_file and converted_audio_path and os.path.exists(converted_audio_path):
            try: os.unlink(converted_audio_path)
            except: pass

if __name__ == "__main__":
    if len(sys.argv) > 3:
        cls_path = sys.argv[4] if len(sys.argv) > 4 else None
        res = separate_audio(sys.argv[1], sys.argv[2], sys.argv[3], cls_path)
        # Clean successful prints
        if res.get("status") == "success":
            print("\n[+] Separation Successful")
            print(f"[+] {len(res.get('stems', {}))} Stems generated and forensic-gated.")
        else:
            print(f"\n[!] Separation Failed: {res.get('message')}")
            
        sys.stdout.write(f"\n[JSON_START]{json.dumps(res)}[JSON_END]\n")
    else:
        err = {"status": "error", "message": "Insufficient arguments"}
        sys.stdout.write(f"\n[JSON_START]{json.dumps(err)}[JSON_END]\n")
    sys.stdout.flush()