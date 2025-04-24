export { };

import { LifeSimulation } from "./gol.js";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

console.log("Checking for WebGPU support...");
const gpu = create([]);
if (!gpu) {
    throw new Error("No appropriate GPU found.");
}
console.log("WebGPU supported.");

console.log("Setting up simulation...");
const gol = await LifeSimulation.create(gpu);
console.log("Simulation setup complete.");

console.log("Beginning warmup...");
const warmupStart = performance.now();
for (let i = 0; i < 100; i++) {
    const encoder = gol.device.createCommandEncoder();
    gol.buildComputePass(encoder, i % 2);
    gol.device.queue.submit([encoder.finish()]);

    // Wait for the work to be done before starting the next one.
    await gol.device.queue.onSubmittedWorkDone();
}
const warmupEnd = performance.now();
console.log(`Warmup took ${(warmupEnd - warmupStart).toFixed(2)}ms`);

let totalDuration = 0;
const runs = 3;
for (let run = 1; run <= runs; run++) {
    console.log(`Beginning benchmark run ${run}...`);
    const start = performance.now();
    for (let i = 0; i < 1_000; i++) {
        const encoder = gol.device.createCommandEncoder();
        gol.buildComputePass(encoder, i % 2);
        gol.device.queue.submit([encoder.finish()]);

        // Wait for the work to be done before starting the next one.
        await gol.device.queue.onSubmittedWorkDone();
    }
    const end = performance.now();

    const durationMs = end - start;
    const speed = 1_000 / (durationMs / 1_000);
    console.log(`Run ${run} took ${durationMs.toFixed(2)}ms (${speed.toFixed(2)} iterations per second)`);
    totalDuration += durationMs;
}

const averageDuration = totalDuration / runs;
const averageSpeed = 1_000 / (averageDuration / 1_000);
console.log(`Average simulation time: ${averageDuration.toFixed(2)}ms (${averageSpeed.toFixed(2)} iterations per second)`);
