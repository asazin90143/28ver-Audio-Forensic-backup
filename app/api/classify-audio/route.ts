import { type NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import os from "os";
import crypto from "crypto";
import { analysisQueue } from "@/lib/job-queue";

export const maxDuration = 900; // Increased to 15 minutes for longer files

async function runPython(scriptName: string, args: string[]): Promise<any> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "scripts", scriptName);
    // Use double quotes for paths that might contain spaces
    const quotedScriptPath = `"${scriptPath}"`;
    const formattedArgs = args.map(arg => arg.startsWith('"') ? arg : `"${arg}"`);

    console.log(`[Forensic] Spawning: python ${quotedScriptPath} ${formattedArgs.join(" ")}`);

    const python = spawn("python", ["-u", quotedScriptPath, ...formattedArgs], {
      shell: true,
      windowsHide: true,
      env: { ...process.env, COLUMNS: "80", PYTHONIOENCODING: "utf-8" }
    });

    let stdout = "";
    let stderr = "";

    const timeout = setTimeout(() => {
      console.error(`[Forensic] Timeout executing ${scriptName}`);
      python.kill();
      reject(new Error(`Timeout executing ${scriptName}. The operation took too long (20 min).`));
    }, 1200000); // 20 Minutes

    let isCapturingJson = false;

    python.stdout.on("data", (data) => {
      const msg = data.toString();
      stdout += msg;

      if (msg.includes("[JSON_START]")) {
        // Print everything BEFORE the JSON marker to the terminal
        const markerIndex = msg.indexOf("[JSON_START]");
        if (markerIndex > 0) {
          process.stdout.write(msg.substring(0, markerIndex));
        }
        isCapturingJson = true;
      }

      // Only print to terminal if we aren't in the middle of a JSON payload
      if (!isCapturingJson) {
        process.stdout.write(msg);
      }
    });

    python.stderr.on("data", (data) => {
      const msg = data.toString();
      stderr += msg;
      const cleaned = msg.replace(/\r/g, "\n");
      const lines = cleaned.split("\n").filter((l: string) => l.trim().length > 0);
      for (const line of lines) {
        process.stderr.write(line + "\n");
      }
    });

    python.on("close", (code) => {
      clearTimeout(timeout);

      if (code !== 0) {
        return reject(new Error(`Python process exited with code ${code}. Error: ${stderr}`));
      }

      try {
        const startMarker = "[JSON_START]";
        const endMarker = "[JSON_END]";
        const startIndex = stdout.indexOf(startMarker);
        const endIndex = stdout.lastIndexOf(endMarker);

        if (startIndex !== -1 && endIndex !== -1) {
          const jsonStr = stdout.substring(startIndex + startMarker.length, endIndex);
          resolve(JSON.parse(jsonStr));
        } else {
          // Fallback: try to find the outermost JSON object
          const start = stdout.indexOf('{');
          const end = stdout.lastIndexOf('}');
          if (start === -1 || end === -1 || end <= start) {
            return reject(new Error(`No JSON found in Python output. Stderr: ${stderr}`));
          }
          const candidate = stdout.substring(start, end + 1);
          const parsed = JSON.parse(candidate);
          // Validate it's a real result object, not random noise
          if (typeof parsed !== "object" || parsed === null) {
            return reject(new Error(`Parsed output is not a valid object. Stderr: ${stderr}`));
          }
          resolve(parsed);
        }
      } catch (e: any) {
        reject(new Error(`Parse error: ${e.message}. Output: ${stdout.substring(0, 500)}...`));
      }
    });

    python.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

export async function POST(request: NextRequest) {
  let tempFilePath = "";
  let audioDataToUse = ""; // Store for fallback

  try {
    const contentType = request.headers.get("content-type") || "";
    let jobID = "";

    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const file = formData.get("audio") as File;
      if (!file) throw new Error("No file uploaded");

      jobID = file.name ? file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase() : `job_${Date.now()}`;

      const tempDir = os.tmpdir();
      tempFilePath = path.join(tempDir, `${jobID}_input.wav`);

      const arrayBuffer = await file.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      fs.writeFileSync(tempFilePath, buffer);

      // Store base64 for fallback
      audioDataToUse = buffer.toString('base64');

    } else {
      // JSON Handling (Base64)
      const { audioData, filename } = await request.json();
      audioDataToUse = audioData; // Store for fallback

      jobID = filename ? filename.replace(/[^a-z0-9]/gi, '_').toLowerCase() : `job_${Date.now()}`;

      const tempDir = os.tmpdir();
      tempFilePath = path.join(tempDir, `${jobID}_input.wav`);
      fs.writeFileSync(tempFilePath, Buffer.from(audioData, 'base64'));
    }

    // 1. Generate an MD5 hash of the file for caching
    const fileBuffer = fs.readFileSync(tempFilePath);
    const fileHash = crypto.createHash('md5').update(fileBuffer).digest('hex');
    const cachePath = path.join(os.tmpdir(), `${fileHash}_cache.json`);

    // 2. Check if we already have this exact file analyzed
    if (fs.existsSync(cachePath)) {
      console.log(`[Cache Hit] Returning existing YAMNet results for ${fileHash}`);
      const cachedResult = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
      return NextResponse.json({
        status: "Success",
        jobID,
        classification: cachedResult,
        cached: true
      });
    }

    // 3. Fallback to external FastAPI if configured
    if (process.env.PYTHON_BACKEND_URL) {
      console.log(`[Decouple] Forwarding classification to ${process.env.PYTHON_BACKEND_URL}/api/classify`);
      const externalFormData = new FormData();
      externalFormData.append("job_id", jobID);
      const blob = new Blob([fileBuffer], { type: "audio/wav" });
      externalFormData.append("audio", blob, "audio.wav");

      const extRes = await fetch(`${process.env.PYTHON_BACKEND_URL}/api/classify`, {
        method: "POST",
        body: externalFormData
      });

      if (!extRes.ok) throw new Error("External FastAPI error");
      const extData = await extRes.json();

      fs.writeFileSync(cachePath, JSON.stringify(extData.classification));

      return NextResponse.json({
        status: "Success",
        jobID,
        classification: extData.classification,
        remote: true
      });
    }

    const outputDir = path.join(process.cwd(), "public", "separated_audio");
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    // Only run Classification (YAMNet) - FAST via Queue
    const classification = await analysisQueue.enqueue(jobID + "_classify", async () => {
      return await runPython("mediapipe_audio_classifier.py", [
        `"${tempFilePath}"`,
        `"${jobID}"`
      ]);
    });

    // Save to cache
    fs.writeFileSync(cachePath, JSON.stringify(classification));

    return NextResponse.json({
      status: "Success",
      jobID,
      classification
    });

  } catch (error: any) {
    console.error("Forensic Engine Error:", error.message);
    return NextResponse.json(
      {
        status: "Error",
        error: error.message,
        details: "The classification engine failed. Check backend logs."
      },
      { status: 500 }
    );
  } finally {
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      try { fs.unlinkSync(tempFilePath); } catch (e) { }
    }
  }
}