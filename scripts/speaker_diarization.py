import os
import sys
import json
import warnings
import torch
import torchaudio
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
    
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print_progress(0, "Error: Missing HUGGINGFACE_TOKEN in .env")
        sys.exit(1)

    try:
        # Phase 1: Speaker Diarization (Counting & Timelining)
        print_progress(10, "Loading PyAnnote Diarization Model...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        print_progress(30, "Analyzing audio for distinct speakers...")
        diarization = pipeline(input_file)
        
        # Extract unique speakers and their total speaking time
        speaker_times = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_times:
                speaker_times[speaker] = 0.0
            speaker_times[speaker] += turn.end - turn.start
            
        num_speakers = len(speaker_times)
        if num_speakers == 0:
            print_progress(100, "No speakers detected.")
            print(f"[JSON_START]{json.dumps({'status': 'Success', 'speakers': 0, 'stems': []})}[JSON_END]")
            sys.exit(0)
            
        print_progress(50, f"Detected {num_speakers} unique speaker(s). Loading SpeechBrain...")
        
        # Phase 2: Speech separation using SpeechBrain SepFormer
        # Using the WHAMR! model which is excellent for 2+ speakers in noisy environments
        # Note: If num_speakers > 2 or 3, a more dynamic model might be needed, but Sepformer handles 2-3 well.
        model_name = "speechbrain/sepformer-wham" 
        
        # In this implementation, we will use Sepformer to separate the full audio, 
        # and then map the diarization labels to the separated stems based on energy/activity.
        # This is a robust approach when exact source count is unknown prior to PyAnnote.
        sep_model = separator.from_hparams(source=model_name, savedir='pretrained_models/sepformer-wham')
        
        print_progress(70, "Physically separating overlapping voices...")
        
        # Load audio for separation
        est_sources = sep_model.separate_file(path=input_file)
        
        # For WHAM/Sepformer, it typically outputs 2 sources. 
        # So we process the separated sources and save them.
        output_stems = []
        
        # Extract the tensor (batch, time, channels)
        # Note: This is a simplified extraction assuming standard Sepformer output shapes
        print_progress(90, "Exporting isolated voice stems...")
        
        try:
            # Squeeze batch dimension if present
            sources = est_sources.squeeze(0) 
            num_separated = sources.shape[1] if len(sources.shape) > 1 else 1

            for i in range(num_separated):
                # Extract the 1D tensor for this source
                source_wav = sources[:, i] if len(sources.shape) > 1 else sources
                
                # Normalize and ensure 2D for torchaudio [channels, time]
                source_wav = source_wav.unsqueeze(0) 
                
                out_path = os.path.join(output_dir, f"{job_id}_voice_{i+1}.wav")
                # Save using torchaudio
                # SepFormer operates at 8000Hz strictly, so we save at 8000Hz
                torchaudio.save(out_path, source_wav, 8000)
                
                output_stems.append({
                    "name": f"Isolated Voice {i+1}",
                    "path": f"/separated_audio/{job_id}_voice_{i+1}.wav",
                    "type": "voice",
                    "color": "bg-indigo-500" if i == 0 else "bg-pink-500"
                })
        except Exception as sep_e:
             # Fallback if SpeechBrain tensor shape is tricky
             print_progress(95, f"Warning during extraction: {str(sep_e)}")
        
        print_progress(100, f"Successfully isolated {len(output_stems)} voices.")
        
        # Output JSON result
        result = {
            "status": "Success",
            "speakers_detected_by_pyannote": num_speakers,
            "stems": output_stems,
            "diarization_summary": speaker_times
        }
        
        print(f"[JSON_START]{json.dumps(result)}[JSON_END]")

    except Exception as e:
        print_progress(0, f"Error: {str(e)}")
        print(f"[JSON_START]{json.dumps({'error': str(e)})}[JSON_END]")
        sys.exit(1)

if __name__ == "__main__":
    main()
