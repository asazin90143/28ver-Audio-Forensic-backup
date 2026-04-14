import os
import sys
import json
import warnings
import numpy as np
import shutil
import tempfile

# Patch os.symlink for Windows (SpeechBrain uses symlinks which require admin on Windows)
_original_symlink = os.symlink
def _safe_symlink(src, dst, *args, **kwargs):
    try:
        _original_symlink(src, dst, *args, **kwargs)
    except OSError:
        # Fall back to copy if symlinks aren't allowed
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
os.symlink = _safe_symlink

# Patch torchaudio backend compatibility for torchaudio >= 2.10
# pyannote.audio and older speechbrain call removed torchaudio APIs
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

import torch
import soundfile as sf

# Patch huggingface_hub: newer versions removed 'use_auth_token' param
# but SpeechBrain 1.0.x still passes it internally
import huggingface_hub
_original_snapshot_download = huggingface_hub.snapshot_download
def _patched_snapshot_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return _original_snapshot_download(*args, **kwargs)
huggingface_hub.snapshot_download = _patched_snapshot_download

_original_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    try:
        return _original_hf_hub_download(*args, **kwargs)
    except Exception as e:
        if "custom.py" in str(kwargs.get("filename", args[1] if len(args) > 1 else "")) and ("404" in str(e) or "Entry Not Found" in str(e)):
            # If SpeechBrain asks for custom.py but it doesn't exist on HF repo, ignore it
            temp_custom = os.path.join(tempfile.gettempdir(), 'speechbrain_empty_custom.py')
            with open(temp_custom, 'w') as f: f.write('')
            return temp_custom
        raise e
huggingface_hub.hf_hub_download = _patched_hf_hub_download

from speechbrain.inference.separation import SepformerSeparation as separator
from pyannote.audio import Pipeline

# Ignore warnings for clean JSON output
warnings.filterwarnings("ignore")

def print_progress(percent: int, text: str):
    """Prints tqdm-style progress so Next.js SSE can parse it"""
    sys.stderr.write(f"{percent}%|{text}\n")
    sys.stderr.flush()

def main():
    if len(sys.argv) < 4:
        print(json.dumps({"error": "Usage: python speaker_diarization.py <input_audio> <output_dir> <job_id>"}))
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    job_id = sys.argv[3]
    min_speakers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    max_speakers = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    os.makedirs(output_dir, exist_ok=True)
    
    # Load .env file explicitly
    try:
        from dotenv import load_dotenv
        # Look for .env in the project root (one level up from scripts directory)
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        # Windows powershell echo/out-file often creates UTF-16LE files (Start bytes: 255 254)
        for enc in ['utf-16le', 'utf-8']:
            try:
                load_dotenv(dotenv_path=env_path, encoding=enc)
                if os.environ.get("HUGGINGFACE_TOKEN"):
                    break # Success!
            except Exception:
                pass
    except ImportError:
        pass

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print_progress(0, "Error: Missing HUGGINGFACE_TOKEN in .env")
        sys.exit(1)
        
    # Explicitly set HF_TOKEN in the environment so the HuggingFace Hub library
    # recognizes it automatically for faster downloads and no warnings.
    os.environ["HF_TOKEN"] = hf_token

    try:
        # Phase 1: Speaker Diarization (Counting & Timelining)
        print_progress(10, "Loading PyAnnote Diarization Model...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )

        try:
            pyannote_ckpt = r"C:\Users\picha\OneDrive\Desktop\CODES\training\models\separation\pyannote_finetune\checkpoint_epoch_20.pt"
            if os.path.exists(pyannote_ckpt):
                print_progress(15, "Injecting custom PyAnnote fine-tuned segmentation weights...")
                custom_state = torch.load(pyannote_ckpt, map_location="cpu", weights_only=False)
                if hasattr(pipeline, '_segmentation') and hasattr(pipeline._segmentation, 'model'):
                    pipeline._segmentation.model.load_state_dict(custom_state.get("model_state", custom_state))
                    print(json.dumps({"progress": 15, "message": "✅ Custom PyAnnote model loaded.", "status": "processing"}))
        except Exception as e:
            print(json.dumps({"progress": 15, "message": f"⚠️ Custom model load failed: {e}", "status": "processing"}))

        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        print_progress(20, "Loading audio file...")
        # Pre-load audio with soundfile to avoid broken torchcodec/torchaudio decoder
        audio_np, sample_rate = sf.read(input_file, dtype='float32')
        # Ensure mono -> shape (1, samples) for PyAnnote
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        waveform = torch.tensor(audio_np).unsqueeze(0).float()
        
        print_progress(30, f"Analyzing audio for distinct speakers (min={min_speakers}, max={max_speakers})...")
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # PyAnnote 3.1 returns a DiarizeOutput wrapper; extract the Annotation object
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization
        
        # Extract unique speakers and their total speaking time
        speaker_times = {}
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            if speaker not in speaker_times:
                speaker_times[speaker] = 0.0
            speaker_times[speaker] += turn.end - turn.start
            
        num_speakers = len(speaker_times)
        if num_speakers == 0:
            print_progress(100, "No speakers detected.")
            print(f"[JSON_START]{json.dumps({'status': 'Success', 'speakers': 0, 'stems': []})}[JSON_END]")
            sys.exit(0)
            
        print_progress(50, f"Detected {num_speakers} unique speaker(s). Analyzing overlaps...")
        
        # =====================================================================
        # Phase 2: Hybrid Diarization-Guided Separation Engine
        # =====================================================================
        sr_target = sample_rate
        duration_samples = waveform.shape[2] if len(waveform.shape) > 2 else waveform.shape[1]
        duration_sec = duration_samples / sr_target
        
        # --- Step 2A: Group segments by speaker ---
        speaker_segments = {}
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((turn.start, turn.end))
        
        speakers_list = list(speaker_segments.keys())
        
        # --- Step 2B: Collision Detection ---
        # Scan all pairs of speakers and find temporal overlaps
        collisions = []  # [(start, end, [speaker_a, speaker_b, ...])]
        
        # Flatten all segments with speaker labels and sort by start time
        all_intervals = []
        for spk, segs in speaker_segments.items():
            for s, e in segs:
                all_intervals.append((s, e, spk))
        all_intervals.sort(key=lambda x: x[0])
        
        # Sweep-line algorithm to find overlapping zones
        for i_idx in range(len(all_intervals)):
            s1, e1, spk1 = all_intervals[i_idx]
            for j_idx in range(i_idx + 1, len(all_intervals)):
                s2, e2, spk2 = all_intervals[j_idx]
                if s2 >= e1:
                    break  # No more possible overlaps with interval i
                if spk1 == spk2:
                    continue  # Same speaker, not a collision
                # Overlap detected
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                if overlap_end - overlap_start >= 0.1:  # At least 100ms overlap
                    # Check if this overlapping zone already exists and merge speakers
                    merged = False
                    for c_idx, (cs, ce, c_spks) in enumerate(collisions):
                        if overlap_start < ce and overlap_end > cs:
                            # Overlapping with existing collision zone — merge
                            collisions[c_idx] = (
                                min(cs, overlap_start),
                                max(ce, overlap_end),
                                list(set(c_spks + [spk1, spk2]))
                            )
                            merged = True
                            break
                    if not merged:
                        collisions.append((overlap_start, overlap_end, [spk1, spk2]))
        
        num_collisions = len(collisions)
        if num_collisions > 0:
            print_progress(55, f"Found {num_collisions} overlapping speech zone(s).")
        else:
            print_progress(55, "No overlapping speech detected. Using direct isolation...")
        
        # --- Step 2C: Lazy SpeechBrain Loading ---
        sep_model = None
        if num_collisions > 0:
            try:
                from speechbrain.inference.separation import SepformerSeparation as separator
                import tempfile
                import torch.nn as nn
                
                model_name = "speechbrain/sepformer-wsj03mix"
                model_dir = "sepformer-wsj03mix"
                print_progress(60, "Loading base SepFormer structure...")
                
                sep_model = separator.from_hparams(
                    source=model_name,
                    savedir=os.path.join(tempfile.gettempdir(), 'speechbrain_models', model_dir)
                )
                
                # --- LOAD CUSTOM FINE-TUNED SEPFORMER+DANN MODEL ---
                custom_sepformer_ckpt = r"C:\Users\picha\OneDrive\Desktop\CODES\training\models\separation\sepformer_dann\checkpoint_epoch_2225.pt"
                if os.path.exists(custom_sepformer_ckpt):
                    print_progress(65, "Injecting custom trained SepFormer+DANN weights (epoch 2225)...")
                    custom_state = torch.load(custom_sepformer_ckpt, map_location="cpu", weights_only=False)
                    if "sep_state" in custom_state and hasattr(sep_model, 'mods'):
                        # Using torch.nn.ModuleDict handles standard dictionary loading mapping exactly to the nested modules
                        nn.ModuleDict(sep_model.mods).load_state_dict(custom_state['sep_state'])
                        print_progress(65, "✅ Custom SepFormer+DANN weights loaded successfully!")
                    else:
                        print_progress(65, "⚠️ Custom model parsing error, using base model.")
                else:
                    print_progress(65, "⚠️ Custom weights file not found!")
                
                # --- PHASE 5: BEATs FRONT-END ADAPTER (Optional) ---
                if os.environ.get("USE_BEATS_FRONTEND", "false").lower() == "true":
                    print_progress(66, "Loading external Microsoft BEATs front-end extractor (2GB)...")
                    try:
                        import sys
                        if "scripts.beats" not in sys.modules:
                            sys.path.insert(0, os.path.join(process.cwd(), "scripts"))
                        
                        from beats.BEATs import BEATs, BEATsConfig
                        import torchaudio.transforms as T
                        
                        beats_ckpt_path = r"C:\Users\picha\OneDrive\Desktop\CODES\training\models\beats\BEATs_iter3_plus_AS2M.pt"
                        beats_ckpt = torch.load(beats_ckpt_path, map_location="cpu", weights_only=False)
                        beats_model = BEATs(BEATsConfig(beats_ckpt["cfg"]))
                        beats_model.load_state_dict(beats_ckpt["model"])
                        beats_model.eval()
                        for p in beats_model.parameters(): p.requires_grad = False
                        
                        adapter = nn.Sequential(
                            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(512, 256), nn.LayerNorm(256)
                        )
                        adapter_path = r"C:\Users\picha\OneDrive\Desktop\CODES\training\models\separation\beats_adapter.pt"
                        adapter.load_state_dict(torch.load(adapter_path, map_location="cpu", weights_only=False))
                        adapter.eval()
                        
                        class BEATsEncoderWrapper(nn.Module):
                            def __init__(self, b_model, b_adapter):
                                super().__init__()
                                self.beats = b_model
                                self.adapter = b_adapter
                                self.resampler = T.Resample(8000, 16000)
                            def forward(self, x):
                                # x is (Batch, Time) at 8kHz
                                x_16k = self.resample(x) if hasattr(self, 'resample') else self.resampler(x)
                                with torch.no_grad():
                                    feats, _ = self.beats.extract_features(x_16k)
                                    adapt = self.adapter(feats) # (B, T, 256)
                                return adapt.transpose(1, 2) # To (B, 256, T) for SpeechBrain masknet
                        
                        sep_model.mods.encoder = BEATsEncoderWrapper(beats_model, adapter)
                        print_progress(66, "✅ Extreme deep-feature acoustic BEATs adapter wired into SepFormer.")
                    except Exception as beats_err:
                        print_progress(66, f"⚠️ Failed to load BEATs frontend: {beats_err}. Falling back to default Conv1D encoder.")
                
            except Exception as sb_err:
                print_progress(65, f"SpeechBrain load failed: {sb_err}. Falling back to direct slicing.")
                sep_model = None
        
        # --- Step 2D: Build collision lookup for fast segment classification ---
        def is_in_collision(time_start, time_end):
            """Check if a time range overlaps with any collision zone. Returns collision or None."""
            for cs, ce, c_spks in collisions:
                if time_start < ce and time_end > cs:
                    return (cs, ce, c_spks)
            return None
        
        def unmix_collision_chunk(collision_start, collision_end, involved_speakers):
            """
            Extract the collision chunk from the master audio, run SepFormer,
            and return a dict mapping speaker -> unmixed numpy array.
            """
            if sep_model is None:
                return None
            
            # Extract chunk from original waveform
            cs_idx = int(collision_start * sr_target)
            ce_idx = int(collision_end * sr_target)
            chunk_np = audio_np[cs_idx:ce_idx].copy()
            
            if len(chunk_np) < 400:
                return None  # Too short for SepFormer
            
            # SepFormer requires mono float32 tensor, shape (1, samples)
            # Resample to 8kHz (SepFormer's training rate)
            chunk_tensor = torch.tensor(chunk_np).unsqueeze(0).float()
            
            if sr_target != 8000:
                import torchaudio.functional as F
                chunk_8k = F.resample(chunk_tensor, orig_freq=sr_target, new_freq=8000)
            else:
                chunk_8k = chunk_tensor
            
            try:
                est_sources = sep_model.separate_batch(chunk_8k)
                # est_sources shape: (1, samples, num_sources) or (batch, samples, sources)
                sources = est_sources.squeeze(0)  # (samples, num_sources)
                
                if len(sources.shape) == 1:
                    sources = sources.unsqueeze(1)
                
                num_channels = min(sources.shape[1], len(involved_speakers))
                
                # Resample each separated channel back to original sample rate
                unmixed = {}
                chunk_duration_samps = ce_idx - cs_idx
                
                for ch in range(num_channels):
                    ch_audio = sources[:, ch].unsqueeze(0)  # (1, samples)
                    
                    if sr_target != 8000:
                        ch_resampled = F.resample(ch_audio, orig_freq=8000, new_freq=sr_target)
                    else:
                        ch_resampled = ch_audio
                    
                    ch_np = ch_resampled.squeeze().cpu().numpy()
                    
                    # Ensure exact length match with original chunk
                    if len(ch_np) > chunk_duration_samps:
                        ch_np = ch_np[:chunk_duration_samps]
                    elif len(ch_np) < chunk_duration_samps:
                        ch_np = np.pad(ch_np, (0, chunk_duration_samps - len(ch_np)))
                    
                    unmixed[ch] = ch_np
                
                # Energy-based speaker assignment:
                # For each involved speaker, find which channel has the most energy
                # during that speaker's known turn within the collision zone
                speaker_to_channel = {}
                used_channels = set()
                
                for spk in involved_speakers:
                    if spk not in speaker_segments:
                        continue
                    
                    # Find this speaker's segments that overlap the collision
                    best_ch = 0
                    best_energy = -1
                    
                    for seg_s, seg_e in speaker_segments[spk]:
                        # Core region where this speaker is active within the collision
                        core_s = max(seg_s, collision_start) - collision_start
                        core_e = min(seg_e, collision_end) - collision_start
                        core_s_idx = int(core_s * sr_target)
                        core_e_idx = int(core_e * sr_target)
                        
                        if core_e_idx <= core_s_idx:
                            continue
                        
                        for ch in range(num_channels):
                            if ch in used_channels:
                                continue
                            ch_slice = unmixed[ch][core_s_idx:core_e_idx]
                            energy = float(np.mean(ch_slice ** 2))
                            if energy > best_energy:
                                best_energy = energy
                                best_ch = ch
                    
                    speaker_to_channel[spk] = best_ch
                    used_channels.add(best_ch)
                
                # Build speaker -> numpy result
                result_map = {}
                for spk in involved_speakers:
                    ch = speaker_to_channel.get(spk, 0)
                    if ch in unmixed:
                        result_map[spk] = unmixed[ch]
                    else:
                        # Fallback: give the raw mixed chunk
                        result_map[spk] = chunk_np[:chunk_duration_samps]
                
                return result_map
            
            except Exception as sep_err:
                print_progress(0, f"SepFormer chunk unmix failed: {sep_err}")
                return None
        
        # --- Step 2E: Assemble per-speaker timelines (clean + unmixed) ---
        print_progress(70, "Assembling speaker timelines...")
        
        # Cache unmixed collision results to avoid re-processing the same collision
        collision_cache = {}
        output_stems = []
        collisions_resolved = 0
        
        spk_idx = 1
        total_speakers = len(speakers_list)
        
        for spk in speakers_list:
            progress_base = 70 + int((spk_idx / total_speakers) * 25)
            print_progress(progress_base, f"Building timeline for {spk.replace('SPEAKER_', 'Speaker ')}...")
            
            spk_y = np.zeros(duration_samples, dtype=np.float32)
            segs = speaker_segments[spk]
            
            for seg_start, seg_end in segs:
                # Add padding for natural extraction
                padded_start = max(0, seg_start - 0.15)
                padded_end = min(duration_sec, seg_end + 0.15)
                
                # Check if this segment collides with another speaker
                collision = is_in_collision(seg_start, seg_end)
                
                if collision is not None and sep_model is not None:
                    # --- COLLISION PATH: Use SpeechBrain to unmix ---
                    c_start, c_end, c_speakers = collision
                    cache_key = (round(c_start, 3), round(c_end, 3))
                    
                    # Unmix this collision (cached)
                    if cache_key not in collision_cache:
                        unmixed_result = unmix_collision_chunk(c_start, c_end, c_speakers)
                        collision_cache[cache_key] = unmixed_result
                        if unmixed_result is not None:
                            collisions_resolved += 1
                    
                    unmixed = collision_cache.get(cache_key)
                    
                    # Split this segment into: pre-collision clean | collision unmixed | post-collision clean
                    # 1. Pre-collision clean part
                    if padded_start < c_start:
                        pre_s = int(padded_start * sr_target)
                        pre_e = int(c_start * sr_target)
                        if pre_e > pre_s and pre_e <= duration_samples:
                            seg_clean = audio_np[pre_s:pre_e].copy()
                            fade_len = min(int(0.05 * sr_target), len(seg_clean) // 2)
                            if fade_len > 0:
                                seg_clean[:fade_len] *= np.linspace(0, 1, fade_len)
                                seg_clean[-fade_len:] *= np.linspace(1, 0, fade_len)
                            spk_y[pre_s:pre_e] += seg_clean
                    
                    # 2. Collision zone (unmixed or fallback to raw)
                    col_s_idx = int(c_start * sr_target)
                    col_e_idx = int(c_end * sr_target)
                    col_len = col_e_idx - col_s_idx
                    
                    if unmixed is not None and spk in unmixed:
                        unmixed_audio = unmixed[spk]
                        # Cross-fade stitch (50ms)
                        fade_len = min(int(0.05 * sr_target), len(unmixed_audio) // 2)
                        if fade_len > 0:
                            unmixed_audio = unmixed_audio.copy()
                            unmixed_audio[:fade_len] *= np.linspace(0, 1, fade_len)
                            unmixed_audio[-fade_len:] *= np.linspace(1, 0, fade_len)
                        actual_len = min(len(unmixed_audio), duration_samples - col_s_idx)
                        spk_y[col_s_idx:col_s_idx + actual_len] += unmixed_audio[:actual_len]
                    else:
                        # Fallback: paste raw audio (same as before — both speakers mixed)
                        if col_e_idx <= duration_samples:
                            seg_raw = audio_np[col_s_idx:col_e_idx].copy()
                            fade_len = min(int(0.05 * sr_target), len(seg_raw) // 2)
                            if fade_len > 0:
                                seg_raw[:fade_len] *= np.linspace(0, 1, fade_len)
                                seg_raw[-fade_len:] *= np.linspace(1, 0, fade_len)
                            spk_y[col_s_idx:col_e_idx] += seg_raw
                    
                    # 3. Post-collision clean part
                    if padded_end > c_end:
                        post_s = int(c_end * sr_target)
                        post_e = int(padded_end * sr_target)
                        if post_e > post_s and post_e <= duration_samples:
                            seg_clean = audio_np[post_s:post_e].copy()
                            fade_len = min(int(0.05 * sr_target), len(seg_clean) // 2)
                            if fade_len > 0:
                                seg_clean[:fade_len] *= np.linspace(0, 1, fade_len)
                                seg_clean[-fade_len:] *= np.linspace(1, 0, fade_len)
                            spk_y[post_s:post_e] += seg_clean
                
                else:
                    # --- CLEAN PATH: Direct array slicing (no overlap) ---
                    start_idx = int(padded_start * sr_target)
                    end_idx = int(padded_end * sr_target)
                    
                    if end_idx <= duration_samples:
                        segment = audio_np[start_idx:end_idx].copy()
                        
                        # Smooth fade
                        fade_len = int(0.05 * sr_target)
                        fade_len = min(fade_len, len(segment) // 2)
                        if fade_len > 0:
                            segment[:fade_len] *= np.linspace(0, 1, fade_len)
                            segment[-fade_len:] *= np.linspace(1, 0, fade_len)
                        
                        spk_y[start_idx:end_idx] += segment
            
            # AGC Normalization
            max_val = np.max(np.abs(spk_y))
            if max_val > 0.05:
                spk_y = spk_y / max_val * 0.9
            
            out_path = os.path.join(output_dir, f"{job_id}_voice_{spk_idx}.wav")
            sf.write(out_path, spk_y, sr_target)
            
            clean_speaker_name = spk.replace("SPEAKER_", "Speaker ")
            output_stems.append({
                "name": clean_speaker_name,
                "path": f"/separated_audio/{job_id}_voice_{spk_idx}.wav",
                "type": "voice",
                "color": "bg-indigo-500" if spk_idx % 2 == 1 else "bg-pink-500"
            })
            spk_idx += 1
        
        # --- Final Report ---
        method_used = "Custom Hybrid (PyAnnote finetune + SepFormer DANN)" if collisions_resolved > 0 else "Custom PyAnnote Direct"
        print_progress(100, f"Done! Isolated {len(output_stems)} voices via {method_used}. Overlapping zones: {num_collisions}")
        
        result = {
            "status": "Success",
            "speakers_detected_by_pyannote": num_speakers,
            "stems": output_stems,
            "diarization_summary": speaker_times,
            "collisions_detected": num_collisions,
            "collisions_resolved": collisions_resolved,
            "method": method_used
        }
        
        print(f"[JSON_START]{json.dumps(result)}[JSON_END]")
        sys.exit(0)

    except Exception as e:
        print_progress(0, f"Error: {str(e)}")
        print(f"[JSON_START]{json.dumps({'error': str(e)})}[JSON_END]")
        sys.exit(1)

if __name__ == "__main__":
    main()
