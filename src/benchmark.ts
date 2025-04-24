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

console.log("Beginning benchmark...");
const start = performance.now();
for (let i = 0; i < 1_000; i++) {
    const encoder = gol.device.createCommandEncoder();
    gol.buildComputePass(encoder, i % 2);
    gol.device.queue.submit([encoder.finish()]);

    // Wait for the work to be done before starting the next one.
    await gol.device.queue.onSubmittedWorkDone();
}
const end = performance.now();

console.log(`Simulation took ${end - start}ms`);
