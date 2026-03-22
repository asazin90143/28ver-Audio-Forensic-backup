import { NextRequest } from "next/server";
import { jobEvents } from "@/lib/event-emitter";

// SSE endpoint to broadcast live progress to frontend
export async function GET(req: NextRequest) {
    const url = new URL(req.url);
    const jobId = url.searchParams.get("jobId");

    if (!jobId) {
        return new Response("Missing jobId", { status: 400 });
    }

    const encoder = new TextEncoder();

    const stream = new ReadableStream({
        start(controller) {
            // Broadcast current progress
            const listener = (progress: { percent: number; text?: string }) => {
                const payload = `data: ${JSON.stringify(progress)}\n\n`;
                try {
                    controller.enqueue(encoder.encode(payload));
                } catch (e) {
                    // Closed by client
                }
            };

            const completeListener = () => {
                const payload = `data: ${JSON.stringify({ percent: 100, text: "Separation Complete", done: true })}\n\n`;
                try {
                    controller.enqueue(encoder.encode(payload));
                    setTimeout(() => { try { controller.close(); } catch(e){} }, 100);
                } catch (e) {
                    // Closed by client
                }
            };

            const errorListener = (errorMsg: string) => {
                const payload = `data: ${JSON.stringify({ error: errorMsg })}\n\n`;
                try {
                    controller.enqueue(encoder.encode(payload));
                    setTimeout(() => { try { controller.close(); } catch(e){} }, 100);
                } catch (e) {
                    // Closed by client
                }
            };

            jobEvents.on(`progress-${jobId}`, listener);
            jobEvents.on(`done-${jobId}`, completeListener);
            jobEvents.on(`error-${jobId}`, errorListener);

            // Keep connection alive with simple comments
            const heartbeat = setInterval(() => {
                try {
                    controller.enqueue(encoder.encode(`: heartbeat\n\n`));
                } catch (e) {
                    clearInterval(heartbeat);
                }
            }, 5000);

            req.signal.addEventListener("abort", () => {
                clearInterval(heartbeat);
                jobEvents.off(`progress-${jobId}`, listener);
                jobEvents.off(`done-${jobId}`, completeListener);
                jobEvents.off(`error-${jobId}`, errorListener);
            });
        }
    });

    return new Response(stream, {
        headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    });
}
