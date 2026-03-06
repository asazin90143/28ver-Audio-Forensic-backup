// Simple in-memory global queue for Node.js (Next.js server)
// This prevents multiple heavy Demucs/YAMNet spawn processes from crashing the server

type JobFunction = () => Promise<any>;

interface QueueJob {
    id: string;
    execute: JobFunction;
    resolve: (value: any) => void;
    reject: (reason: any) => void;
}

class JobQueue {
    private queue: QueueJob[] = [];
    private processing: boolean = false;
    private maxConcurrency: number;
    private activeJobs: number = 0;

    constructor(maxConcurrency: number = 1) {
        this.maxConcurrency = maxConcurrency;
    }

    public async enqueue(id: string, execute: JobFunction): Promise<any> {
        return new Promise((resolve, reject) => {
            this.queue.push({ id, execute, resolve, reject });
            this.processQueue();
        });
    }

    public getQueuePosition(id: string): number {
        return this.queue.findIndex(job => job.id === id) + 1; // 0 means active/not in queue, 1 means first in line
    }

    private async processQueue() {
        if (this.activeJobs >= this.maxConcurrency || this.queue.length === 0) {
            return;
        }

        this.activeJobs++;
        const job = this.queue.shift();

        if (job) {
            try {
                console.log(`[JobQueue] Starting job ${job.id}. Active: ${this.activeJobs}, Pending in queue: ${this.queue.length}`);
                const result = await job.execute();
                job.resolve(result);
            } catch (error) {
                console.error(`[JobQueue] Job ${job.id} failed:`, error);
                job.reject(error);
            } finally {
                this.activeJobs--;
                console.log(`[JobQueue] Finished job ${job.id}. Active: ${this.activeJobs}, Pending in queue: ${this.queue.length}`);
                this.processQueue();
            }
        }
    }
}

// Global instance to persist across HMR (Hot Module Replacement) during dev
// In production, this persists as long as the Node process lives
const globalForQueue = global as unknown as { jobQueue: JobQueue };
export const analysisQueue = globalForQueue.jobQueue || new JobQueue(1);

if (process.env.NODE_ENV !== "production") {
    globalForQueue.jobQueue = analysisQueue;
}
