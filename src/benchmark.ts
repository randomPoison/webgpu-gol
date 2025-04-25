export { };

import { LifeSimulation } from "./gol.js";
import { create, globals } from "webgpu";

const WARMUP_ITERATIONS = 100;
const BENCHMARK_ITERATIONS = 1_000;
const TEST_RUNS = 3;

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
for (let i = 0; i < WARMUP_ITERATIONS; i++) {
    const encoder = gol.device.createCommandEncoder();
    gol.buildComputePass(encoder, i % 2);
    gol.device.queue.submit([encoder.finish()]);

    // Wait for the work to be done before starting the next one.
    await gol.device.queue.onSubmittedWorkDone();
}
const warmupEnd = performance.now();
console.log(`Warmup took ${(warmupEnd - warmupStart).toFixed(2)}ms`);

let totalDuration = 0;
for (let run = 1; run <= TEST_RUNS; run++) {
    console.log(`Beginning benchmark run ${run}...`);
    const start = performance.now();

    const encoder = gol.device.createCommandEncoder();
    for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
        gol.buildComputePass(encoder, i % 2);
    }
    gol.device.queue.submit([encoder.finish()]);

    // Wait for the work to be done before starting the next one.
    await gol.device.queue.onSubmittedWorkDone();

    const end = performance.now();

    const durationMs = end - start;
    const speed = BENCHMARK_ITERATIONS / (durationMs / 1_000);
    console.log(`Run ${run} took ${durationMs.toFixed(2)}ms (${speed.toFixed(2)} iterations per second)`);
    totalDuration += durationMs;
}

const averageDuration = totalDuration / TEST_RUNS;
const averageSpeed = BENCHMARK_ITERATIONS / (averageDuration / 1_000);
console.log(`Average simulation time: ${averageDuration.toFixed(2)}ms (${averageSpeed.toFixed(2)} iterations per second)`);
