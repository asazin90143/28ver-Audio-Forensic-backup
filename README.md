# Forensic Sonar V4 - Audio Forensic Detector

Advanced Audio Analysis & Target Signal Isolation System.
This is a comprehensive full-stack application built with Next.js and Python that performs deep forensic analysis, classification, and separation of audio signals.

## 🚀 Features

- **Live Sonar Visualization:** Real-time 2D radar and 3D terrain mapping of audio signals.
- **Deep Signal Classification:** Utilizes Google's YAMNet (via MediaPipe) to classify over 500+ distinct sound events with high precision and confidence scoring. Deduplicates and filters ambient noise.
- **Audio Deconstruction (Stems):** Uses Meta's HTDemucs AI model to isolate and extract specific targets (e.g., Human Voice, Background Noise, Sirens) into high-fidelity stems.
- **Advanced Speaker Diarization:** Integrates PyAnnote 3.1 to identify distinct speakers, create a labeled forensic timeline, and calculate speaker confidence scores.
- **Target Voice Isolation:** Chains Demucs vocals to SpeechBrain's SepFormer model to extract individual, clean voices from overlapping conversations.
- **Live Progress Streaming:** Implements a real-time SSE (Server-Sent Events) pipeline to provide live percentage-based feedback during heavy AI workloads.

## 🛠️ Technology Stack

### Frontend (Node.js)
- **Framework**: Next.js 14, React 18
- **Styling**: Tailwind CSS, Radix UI components
- **Visualization**: HTML5 Canvas (2D Radar/3D Terrain), WaveSurfer.js
- **Streaming**: Server-Sent Events (SSE) for live job tracking

### Backend Engine (Python)
- **Audio Processing**: Librosa, SciPy, SoundFile
- **Classification AI**: TensorFlow, MediaPipe (YAMNet)
- **Diarization AI**: PyAnnote Audio 3.1 (State-of-the-art Speaker Identification)
- **Separation AI**: Meta HTDemucs & SpeechBrain SepFormer (Blind Source Separation)
- **Media Transcoding**: FFmpeg

## ⚙️ Prerequisites

Before installing the application, ensure your system has the following requirements:

1. **Node.js** (v18 or higher) & **npm**
2. **Python** (v3.10 or higher)
3. **FFmpeg** (Must be installed and strictly added to your system's `PATH`)
4. **Hugging Face Account** (Required for the Diarization engine access)

## 💻 Local Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/asazin90143/28ver-Audio-Forensic-main
cd 28ver-Audio-Forensic-main
```

**2. Install Node.js Dependencies**
```bash
npm install
```

**3. Install Python Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up Environment Variables**
Create a `.env` file in the root directory:
```env
HUGGINGFACE_TOKEN=your_hf_token_here
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```
> [!IMPORTANT]
> To use **Isolate Voices**, you must accept the terms for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on Hugging Face and provide an access token in the `.env` file.

**5. Start the Development Server**
```bash
npm run dev
```

## 📁 System Architecture Notes

- **Job Queuing**: All heavy AI tasks (Separation/Diarization) are managed by a custom FIFO Job Queue (`lib/job-queue.ts`) to prevent server CPU exhaustion.
- **Live Feedback**: Progress is streamed from Python to the UI via an SSE router (`app/api/progress/route.ts`).
- **Forensic Chaining**: The "Isolate Voices" pipeline automatically feeds the Demucs `vocals` stem into PyAnnote for exponentially higher accuracy compared to raw audio processing.
- **Automatic Cleanup**: A disk-cleanup protocol runs on separation routes to delete temp files and stems older than 1 hour.

## 📝 License

MIT License - see LICENSE file for details.
