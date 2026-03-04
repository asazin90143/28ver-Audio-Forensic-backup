# Forensic Sonar V4 - Audio Forensic Detector

Advanced Audio Analysis & Target Signal Isolation System.
This is a comprehensive full-stack application built with Next.js and Python that performs deep forensic analysis, classification, and separation of audio signals.

## 🚀 Features

- **Live Sonar Visualization:** Real-time 2D radar and 3D terrain mapping of audio signals.
- **Deep Signal Classification:** Utilizes Google's YAMNet (via MediaPipe) to classify over 500+ distinct sound events with high precision and confidence scoring. Deduplicates and filters ambient noise.
- **Audio Deconstruction (Stems):** Uses Meta's HTDemucs AI model to isolate and extract specific targets (e.g., Human Voice, Background Noise, Sirens, Music) from complex audio environments.
- **Enhanced Analysis View:** Generates detailed breakdowns including amplitude, decibels, estimated distance, frequency charts, and time-stamped visual representations.
- **Reporting & Cloud Storage:** Export comprehensive PDF forensic reports, and optionally save enhanced analysis metadata to a Supabase PostgreSQL database.
- **Multi-Input Support:** Supports live microphone streaming and pre-recorded audio file uploads (MP3, WAV, M4A, OGG).

## 🛠️ Technology Stack

### Frontend (Node.js)
- **Framework**: Next.js 14, React 18
- **Styling**: Tailwind CSS, Radix UI components
- **Visualization**: HTML5 Canvas (2D/3D), WaveSurfer.js
- **Environment**: TypeScript

### Backend Engine (Python)
- **Audio Processing**: Librosa, SciPy
- **Classification AI**: TensorFlow, MediaPipe (YAMNet)
- **Separation AI**: Demucs (`htdemucs` model) by Facebook Research
- **Media Transcoding**: FFmpeg

## ⚙️ Prerequisites

Before installing the application, ensure your system has the following requirements:

1. **Node.js** (v18 or higher) & **npm**
2. **Python** (v3.9 or higher)
3. **FFmpeg** (Must be installed and strictly added to your system's `PATH` variables)
4. **Git**

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
# It is highly recommended to use a virtual environment
pip install -r requirements.txt
```

**4. Set up Environment Variables**
Create a `.env` or `.env.local` file in the root directory:
```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```
*(Supabase credentials are required only if you use the database saving feature, otherwise the app will run locally without a database).*

**5. Start the Development Server**
```bash
npm run dev
```

**6. Access the Application**
Open [http://localhost:3000](http://localhost:3000) in your web browser.

## 📁 System Architecture Notes

- The frontend interacts with Python scripts via Next.js API Routes (`app/api/classify-audio/route.ts` and `app/api/separate-audio/route.ts`).
- `classify-audio` spawns `scripts/mediapipe_audio_classifier.py` for ultra-fast event detection.
- `separate-audio` spawns `scripts/audio_separator.py` (which requires FFmpeg) to extract stems into `public/separated_audio`.
- An automatic disk-cleanup protocol runs on the Next.js API route to delete separation stems that are older than 1 hour to prevent disk-space exhaustion.

## 📝 License

MIT License - see LICENSE file for details.
