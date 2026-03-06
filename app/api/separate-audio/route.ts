import { type NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import os from "os";
import { analysisQueue } from "@/lib/job-queue";

export const maxDuration = 900;

async function runPython(scriptName: string, args: string[]): Promise<any> {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(process.cwd(), "scripts", scriptName);
        const quotedScriptPath = `"${scriptPath}"`;
        const formattedArgs = args.map(arg => arg.startsWith('"') ? arg : `"${arg}"`);

        console.log(`[Forensic-Separation] Spawning: python ${quotedScriptPath} ${formattedArgs.join(" ")}`);

        const python = spawn("python", ["-u", quotedScriptPath, ...formattedArgs], {
            shell: true,
            windowsHide: true,
            env: { ...process.env, COLUMNS: "80", PYTHONIOENCODING: "utf-8" }
        });

        let stdout = "";
        let stderr = "";

        const timeout = setTimeout(() => {
            python.kill();
            reject(new Error(`Timeout executing ${scriptName}. Engine took too long.`));
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
            // Clean up tqdm progress bars: replace \r with \n so each update is its own line
            const cleaned = msg.replace(/\r/g, "\n");
            const lines = cleaned.split("\n").filter((l: string) => l.trim().length > 0);
            for (const line of lines) {
                process.stderr.write(line + "\n");
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
                    // Fallback: try to find the outermost JSON object
                    const start = stdout.indexOf('{');
                    const end = stdout.lastIndexOf('}');
                    if (start === -1 || end === -1 || end <= start) {
                        return reject(new Error(`No JSON found in Python output. Stderr: ${stderr}`));
                    }
                    const candidate = stdout.substring(start, end + 1);
                    const parsed = JSON.parse(candidate);
                    if (typeof parsed !== "object" || parsed === null) {
                        return reject(new Error(`Parsed output is not a valid object. Stderr: ${stderr}`));
                    }
                    resolve(parsed);
                }
            } catch (e: any) {
                reject(new Error(`Parse error: ${e.message}`));
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
    let classificationPath = "";

    try {
        const formData = await request.formData();
        const file = formData.get("audio") as File;
        const classificationJson = formData.get("classification") as string;

        if (!file || !classificationJson) throw new Error("Missing audio or classification data");

        const jobID = file.name ? file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase() : `sep_${Date.now()}`;
        const tempDir = os.tmpdir();

        // 1. Save Audio to Temp
        tempFilePath = path.join(tempDir, `${jobID}_sep_input.wav`);
        const arrayBuffer = await file.arrayBuffer();
        fs.writeFileSync(tempFilePath, Buffer.from(arrayBuffer));

        // 2. Save Classification to Temp (so python can read it)
        classificationPath = path.join(tempDir, `${jobID}_sep_class.json`);
        fs.writeFileSync(classificationPath, classificationJson);

        const outputDir = path.join(process.cwd(), "public", "separated_audio");
        if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

        // Cleanup: Remove separation folders older than 1 hour
        try {
            const MAX_AGE_MS = 60 * 60 * 1000; // 1 hour
            const now = Date.now();
            const subDirs = ["htdemucs", "generated"];
            for (const sub of subDirs) {
                const subPath = path.join(outputDir, sub);
                if (fs.existsSync(subPath)) {
                    const entries = fs.readdirSync(subPath);
                    for (const entry of entries) {
                        const entryPath = path.join(subPath, entry);
                        try {
                            const stat = fs.statSync(entryPath);
                            if (stat.isDirectory() && (now - stat.mtimeMs) > MAX_AGE_MS) {
                                fs.rmSync(entryPath, { recursive: true, force: true });
                                console.log(`[Cleanup] Removed old folder: ${entry}`);
                            }
                        } catch (e) { /* skip individual errors */ }
                    }
                }
            }
        } catch (e) { /* cleanup is best-effort, don't fail the request */ }

        // 3. Run Separation via Job Queue
        const separation = await analysisQueue.enqueue(jobID, async () => {
            return await runPython("audio_separator.py", [
                `"${tempFilePath}"`,
                `"${outputDir}"`,
                `"${jobID}"`,
                `"${classificationPath}"`
            ]);
        });

        return NextResponse.json({
            status: "Success",
            stems: separation.stems,
            debug: separation.debug
        });

    } catch (error: any) {
        console.error("Separation Error:", error.message);
        return NextResponse.json({ status: "Error", error: error.message }, { status: 500 });
    } finally {
        if (tempFilePath && fs.existsSync(tempFilePath)) try { fs.unlinkSync(tempFilePath); } catch (e) { }
        if (classificationPath && fs.existsSync(classificationPath)) try { fs.unlinkSync(classificationPath); } catch (e) { }
    }
}
