from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os
import shutil
import json
import tempfile

app = FastAPI(title="Forensic Audio Server V4", description="Backend Engine for Demucs & YAMNet separation/classification")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
TEMP_DIR = tempfile.gettempdir()

def run_script(script_name: str, args: list):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        raise HTTPException(status_code=500, detail=f"Script not found: {script_path}")
    
    cmd = ["python", "-u", script_path] + args
    print(f"[FastAPI] Running: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["COLUMNS"] = "80"
    env["PYTHONIOENCODING"] = "utf-8"
    
    # We use subprocess.run to wait for the result
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )
    
    # Extract JSON between markers
    stdout = process.stdout
    stderr = process.stderr
    
    if process.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Execution Failed:\n{stderr}")
    
    start_marker = "[JSON_START]"
    end_marker = "[JSON_END]"
    
    start_idx = stdout.find(start_marker)
    end_idx = stdout.rfind(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        json_str = stdout[start_idx + len(start_marker):end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"JSON Parse Error: {str(e)}\n{json_str}")
    
    # Fallback outer JSON block
    start = stdout.find('{')
    end = stdout.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        candidate = stdout[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            pass
            
    raise HTTPException(status_code=500, detail=f"No JSON found in stdout. Output: {stdout[:500]}")


@app.post("/api/classify")
async def classify_audio(job_id: str = Form(...), audio: UploadFile = File(...)):
    # Save audio temporarily
    temp_wav = os.path.join(TEMP_DIR, f"{job_id}_classify.wav")
    try:
        with open(temp_wav, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Run YAMNet script
        result = run_script("mediapipe_audio_classifier.py", [temp_wav, job_id])
        return {"status": "Success", "jobID": job_id, "classification": result}
        
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


@app.post("/api/separate")
async def separate_audio(
    job_id: str = Form(...), 
    classification: str = Form(...), 
    audio: UploadFile = File(...)
):
    temp_wav = os.path.join(TEMP_DIR, f"{job_id}_separate.wav")
    class_json = os.path.join(TEMP_DIR, f"{job_id}_class.json")
    output_dir = os.path.join(BASE_DIR, "public", "separated_audio")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save audio
        with open(temp_wav, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Save classification JSON mapping
        with open(class_json, "w", encoding="utf-8") as f:
            f.write(classification)
            
        # Run Demucs Script
        result = run_script("audio_separator.py", [temp_wav, output_dir, job_id, class_json])
        return {"status": "Success", "stems": result.get("stems", [])}
        
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)
        if os.path.exists(class_json): os.remove(class_json)

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
