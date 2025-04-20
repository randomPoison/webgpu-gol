export { };

import { GRID_SIZE, LifeSimulation } from "./gol";

export const UPDATE_INTERVAL_MS = 1000 / 60;

const canvas = document.querySelector("canvas");
if (!canvas) {
    throw new Error("No canvas found in the document.");
}

// Check for WebGPU support.
if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
}

let gol = await LifeSimulation.create(navigator.gpu);

// Setup the canvas for WebGPU.
const canvasContext = canvas.getContext("webgpu");
if (!canvasContext) {
    throw new Error("No WebGPU context found.");
}

const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
canvasContext.configure({
    device: gol.device,
    format: canvasFormat,
});

// Setup the vertex buffer.
const vertices = new Float32Array([
    -0.8, -0.8,
    0.8, -0.8,
    0.8, 0.8,

    -0.8, -0.8,
    0.8, 0.8,
    -0.8, 0.8,
]);
const vertexBuffer = gol.device.createBuffer({
    label: "Cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
gol.device.queue.writeBuffer(vertexBuffer, 0, vertices);

const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
        format: "float32x2" as GPUVertexFormat,
        offset: 0,
        shaderLocation: 0, // Position
    }],
};

// Setup the shaders.
const cellShaderModule = gol.device.createShaderModule({
    label: "Cell shader",
    code: `
        struct VertexInput {
          @location(0) pos: vec2f,
          @builtin(instance_index) instance: u32,
        };

        struct VertexOutput {
          @builtin(position) pos: vec4f,
          @location(0) cell: vec2f,
        };

        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellState: array<u32>;

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
          let i = f32(input.instance);

          let x = i % grid.x;
          let y = floor(i / grid.x);
          let cell = vec2f(x, y);

          let state = f32(cellState[input.instance]);

          let cellOffset = cell / grid * 2;
          let gridPos = (input.pos * state + 1) / grid - 1 + cellOffset;

          var output: VertexOutput;
          output.pos = vec4f(gridPos, 0, 1);
          output.cell = cell;
          return output;
        }

        @fragment
        fn fragmentMain(@location(0) cell: vec2f) -> @location(0) vec4f {
          let c = cell / grid;
          return vec4f(c, 1 - c.y, 1);
        }
      `
});

const cellPipeline = gol.device.createRenderPipeline({
    label: "Cell pipeline",
    layout: gol.pipelineLayout,
    vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout],
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{
            format: canvasFormat,
        }],
    }
});

let step = 0;
function updateAndRender() {
    const encoder = gol.device.createCommandEncoder();

    gol.buildComputePass(encoder, step);

    // Update the step between the render passes such that the output of the
    // computer pass becomes the input of the render pass.
    step++;

    // Start a render pass.
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: canvasContext!.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: [1, 1, 1, 1],
            storeOp: "store",
        }]
    });

    // Draw the grid.
    pass.setPipeline(cellPipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, gol.bindGroups[step % 2]);
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);

    // Finish and submit the render pass.
    pass.end();
    gol.device.queue.submit([encoder.finish()]);
}

// Run update and render loop.
setInterval(updateAndRender, UPDATE_INTERVAL_MS);
