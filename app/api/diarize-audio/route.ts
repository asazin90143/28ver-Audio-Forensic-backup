import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import os from "os";
import { analysisQueue } from "@/lib/job-queue";
import { jobEvents } from "@/lib/event-emitter";

export const maxDuration = 900; // Allow 15 mins for heavy Diarization + SepFormer

async function runPython(scriptName: string, args: string[], jobId?: string): Promise<any> {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(process.cwd(), "scripts", scriptName);
        const quotedScriptPath = `"${scriptPath}"`;
        const formattedArgs = args.map(arg => arg.startsWith('"') ? arg : `"${arg}"`);

        console.log(`[Forensic-Diarization] Spawning: python ${quotedScriptPath} ${formattedArgs.join(" ")}`);

        const python = spawn("python", ["-u", quotedScriptPath, ...formattedArgs], {
            shell: true,
            windowsHide: true,
            env: { ...process.env, COLUMNS: "80", PYTHONIOENCODING: "utf-8" }
        });

        let stdout = "";
        let stderr = "";

        const timeout = setTimeout(() => {
            python.kill();
            reject(new Error(`Timeout executing ${scriptName}. Voice Isolation took too long.`));
        }, 1200000); // 20 Mins

        let isCapturingJson = false;

        python.stdout.on("data", (data) => {
            const msg = data.toString();
            stdout += msg;

            if (msg.includes("[JSON_START]")) {
                const markerIndex = msg.indexOf("[JSON_START]");
                if (markerIndex > 0) process.stdout.write(msg.substring(0, markerIndex));
                isCapturingJson = true;
            }

            if (!isCapturingJson) process.stdout.write(msg);
        });

        python.stderr.on("data", (data) => {
            const msg = data.toString();
            stderr += msg;
            
            const cleaned = msg.replace(/\r/g, "\n");
            const lines = cleaned.split("\n").filter((l: string) => l.trim().length > 0);
            
            for (const line of lines) {
                process.stderr.write(line + "\n");
                
                // Parse tqdm percent
                if (jobId) {
                    const match = line.match(/(\d+)%\|(.*)/);
                    if (match) {
                        const percent = parseInt(match[1], 10);
                        const text = match[2] ? match[2].trim() : "Analyzing Voices...";
                        jobEvents.emit(`progress-${jobId}`, { percent, text });
                    }
                }
            }
        });

        python.on("close", (code) => {
            clearTimeout(timeout);
            if (code !== 0) return reject(new Error(`Exit ${code}: ${stderr}`));
            try {
                const startMarker = "[JSON_START]";
                const endMarker = "[JSON_END]";
                const startIndex = stdout.indexOf(startMarker);
                const endIndex = stdout.lastIndexOf(endMarker);

                if (startIndex !== -1 && endIndex !== -1) {
                    const jsonStr = stdout.substring(startIndex + startMarker.length, endIndex);
                    resolve(JSON.parse(jsonStr));
                } else {
                    const start = stdout.indexOf('{');
                    const end = stdout.lastIndexOf('}');
                    if (start === -1 || end === -1 || end <= start) {
                        return reject(new Error(`No JSON found in Python output.`));
                    }
                    resolve(JSON.parse(stdout.substring(start, end + 1)));
                }
                
                if (jobId) jobEvents.emit(`done-${jobId}`);
            } catch (e: any) {
                if (jobId) jobEvents.emit(`error-${jobId}`, `Parse error: ${e.message}`);
                reject(new Error(`Parse error: ${e.message}`));
            }
        });

        python.on("error", (err) => {
            clearTimeout(timeout);
            if (jobId) jobEvents.emit(`error-${jobId}`, err.message);
            reject(err);
        });
    });
}

export async function POST(request: NextRequest) {
    let tempFilePath = "";

    try {
        const formData = await request.formData();
        const file = formData.get("audio") as File;
        if (!file) throw new Error("No file uploaded");

        const filename = file.name || "audio.wav";
        const jobID = filename.replace(/[^a-z0-9]/gi, '_').toLowerCase();

        // 1. Save uploaded file to temp directory
        const tempDir = os.tmpdir();
        tempFilePath = path.join(tempDir, `${jobID}_input.wav`);
        
        const arrayBuffer = await file.arrayBuffer();
        fs.writeFileSync(tempFilePath, Buffer.from(arrayBuffer));

        const outputDir = path.join(process.cwd(), "public", "separated_audio");
        if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

        // 2. Add to Queue and Run Voice Isolation Pipeline
        const isolation = await analysisQueue.enqueue(jobID + "_diarization", async () => {
            return await runPython("speaker_diarization.py", [
                `"${tempFilePath}"`,
                `"${outputDir}"`,
                `"${jobID}"`
            ], jobID);
        });

        // Add correct absolute URLs for the frontend audio player
        const processedStems = isolation.stems?.map((stem: any) => ({
            name: stem.name,
            url: stem.path, // The script returns /separated_audio/...
            type: stem.type,
        })) || [];

        return NextResponse.json({
            status: "Success",
            speakers: isolation.speakers_detected_by_pyannote,
            stems: processedStems,
            diarization: isolation.diarization_summary
        });

    } catch (error: any) {
        console.error("Diarization Error:", error.message);
        return NextResponse.json({ status: "Error", error: error.message }, { status: 500 });
    } finally {
        // Cleanup temp files immediately
        if (tempFilePath && fs.existsSync(tempFilePath)) {
            try { fs.unlinkSync(tempFilePath); } catch (e) { }
        }
    }
}
