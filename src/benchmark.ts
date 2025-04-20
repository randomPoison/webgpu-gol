export { };

import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const gpu = create([]);

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
}

const device = await adapter.requestDevice();
