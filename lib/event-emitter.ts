import { EventEmitter } from "events";

// Global event emitter to persist across HMR in development
const globalForEmitter = global as unknown as { jobEmitter: EventEmitter };
export const jobEvents = globalForEmitter.jobEmitter || new EventEmitter();

// Increase max listeners since we might have many active connections
jobEvents.setMaxListeners(100);

if (process.env.NODE_ENV !== "production") {
    globalForEmitter.jobEmitter = jobEvents;
}
