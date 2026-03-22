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
        
        print_progress(30, "Analyzing audio for distinct speakers...")
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        
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
            
        print_progress(50, f"Detected {num_speakers} unique speaker(s). Loading SpeechBrain...")
        
        # Phase 2: Dynamic Speech separation based on speaker count
        if num_speakers <= 2:
            model_name = "speechbrain/sepformer-wsj02mix"
            print_progress(70, f"Detected {num_speakers} speakers. Running standard 2-channel SepFormer...")
            sep_model = separator.from_hparams(source=model_name, savedir=os.path.join(tempfile.gettempdir(), 'speechbrain_models', 'sepformer-wsj02mix'))
        elif num_speakers == 3:
            model_name = "speechbrain/sepformer-wsj03mix"
            print_progress(70, f"Detected 3 speakers. Running advanced 3-channel SepFormer...")
            sep_model = separator.from_hparams(source=model_name, savedir=os.path.join(tempfile.gettempdir(), 'speechbrain_models', 'sepformer-wsj03mix'))
        else:
            model_name = "speechbrain/sepformer-wsj02mix" # The 2-channel handles the isolated 2-person clips
            print_progress(70, f"Detected {num_speakers} speakers (>3). Engaging Temporal Sequential Isolation...")
            sep_model = separator.from_hparams(source=model_name, savedir=os.path.join(tempfile.gettempdir(), 'speechbrain_models', 'sepformer-wsj02mix'))

        # Resample to 8kHz for SepFormer
        if sample_rate != 8000:
            import torchaudio.functional as F
            waveform_8k = F.resample(waveform, orig_freq=sample_rate, new_freq=8000)
        else:
            waveform_8k = waveform
            
        output_stems = []

        if num_speakers <= 3:
            # Standard Separation
            print_progress(80, "Physically separating overlapping voices...")
            est_sources = sep_model.separate_batch(waveform_8k)
            
            print_progress(90, "Exporting isolated voice stems...")
            try:
                sources = est_sources.squeeze(0) 
                num_separated = sources.shape[1] if len(sources.shape) > 1 else 1

                for i in range(num_separated):
                    source_wav = sources[:, i] if len(sources.shape) > 1 else sources
                    source_wav = source_wav.unsqueeze(0) 
                    
                    out_path = os.path.join(output_dir, f"{job_id}_voice_{i+1}.wav")
                    wav_numpy = source_wav.squeeze().cpu().numpy()
                    
                    # Optional AGC Normalization for safety
                    max_val = np.max(np.abs(wav_numpy))
                    if max_val > 0.0: wav_numpy = wav_numpy / max_val * 0.9
                    
                    sf.write(out_path, wav_numpy, 8000)
                    
                    output_stems.append({
                        "name": f"Isolated Voice {i+1}",
                        "path": f"/separated_audio/{job_id}_voice_{i+1}.wav",
                        "type": "voice",
                        "color": "bg-indigo-500" if i == 0 else "bg-pink-500"
                    })
            except Exception as sep_e:
                 print_progress(95, f"Warning during extraction: {str(sep_e)}")
                 
        else:
            # TEMPORAL SEQUENTIAL ISOLATION (4+ Speakers)
            sr_8k = 8000
            duration_samples = waveform_8k.shape[1]
            
            stitched_tracks = {speaker: np.zeros(duration_samples, dtype=np.float32) for speaker in speaker_times.keys()}
            
            total_turns = sum(1 for _ in annotation.itertracks())
            turn_idx = 0
            
            # Use PyAnnote generator to target exact timestamps
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                turn_idx += 1
                if turn_idx % max(1, total_turns // 5) == 0:
                    print_progress(70 + int((turn_idx / total_turns) * 20), f"Sequential Isolation: Turn {turn_idx}/{total_turns}...")
                
                # Context Window Padding (0.5s pre/post-roll)
                start_sec = max(0.0, turn.start - 0.5)
                end_sec = min(duration_samples / sr_8k, turn.end + 0.5)
                
                start_samp = int(start_sec * sr_8k)
                end_samp = int(end_sec * sr_8k)
                
                if end_samp - start_samp < 400: continue
                    
                chunk = waveform_8k[:, start_samp:end_samp]
                
                # Separate just this chunk
                try:
                    chunk_sources = sep_model.separate_batch(chunk).squeeze(0) # shape: [time, 2]
                except Exception:
                    continue
                
                # Energy calculation to identify the target speaker stem
                core_start = int((turn.start - start_sec) * sr_8k)
                core_end = int((turn.end - start_sec) * sr_8k)
                core_start = max(0, min(core_start, chunk_sources.shape[0]))
                core_end = max(0, min(core_end, chunk_sources.shape[0]))
                
                if core_end > core_start:
                    energy_0 = torch.mean(chunk_sources[core_start:core_end, 0] ** 2)
                    energy_1 = torch.mean(chunk_sources[core_start:core_end, 1] ** 2)
                    best_idx = 0 if energy_0 > energy_1 else 1
                else:
                    best_idx = 0
                    
                best_audio = chunk_sources[:, best_idx].cpu().numpy()
                
                # Cross-Fade Stitching (50ms) to prevent boundary pops 
                fade_len = int(0.05 * sr_8k)
                fade_len = min(fade_len, len(best_audio) // 2)
                if fade_len > 0:
                    fade_in = np.linspace(0, 1, fade_len)
                    fade_out = np.linspace(1, 0, fade_len)
                    best_audio[:fade_len] *= fade_in
                    best_audio[-fade_len:] *= fade_out
                
                slice_len = len(best_audio)
                if start_samp + slice_len > duration_samples:
                    slice_len = duration_samples - start_samp
                    best_audio = best_audio[:slice_len]
                    
                # Blend snippet into the master multi-track matrix
                stitched_tracks[speaker][start_samp:start_samp+slice_len] += best_audio
            
            print_progress(95, "Applying AGC Volume Normalization to assembled timelines...")
            i = 0
            for speaker, track in stitched_tracks.items():
                # Post-Stitch Volume Normalization (AGC)
                max_val = np.max(np.abs(track))
                if max_val > 0.02: 
                    track = track / max_val * 0.9
                
                out_path = os.path.join(output_dir, f"{job_id}_voice_{i+1}.wav")
                sf.write(out_path, track, 8000)
                
                output_stems.append({
                    "name": f"Isolated Voice {i+1} ({speaker})",
                    "path": f"/separated_audio/{job_id}_voice_{i+1}.wav",
                    "type": "voice",
                    "color": "bg-indigo-500" if i == 0 else "bg-pink-500"
                })
                i += 1
        
        print_progress(100, f"Successfully isolated {len(output_stems)} voices.")
        
        # Output JSON result
        result = {
            "status": "Success",
            "speakers_detected_by_pyannote": num_speakers,
            "stems": output_stems,
            "diarization_summary": speaker_times
        }
        
        print(f"[JSON_START]{json.dumps(result)}[JSON_END]")
        sys.exit(0)

    except Exception as e:
        print_progress(0, f"Error: {str(e)}")
        print(f"[JSON_START]{json.dumps({'error': str(e)})}[JSON_END]")
        sys.exit(1)

if __name__ == "__main__":
    main()
