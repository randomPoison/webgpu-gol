export { };

const GRID_SIZE = 1024;
const UPDATE_INTERVAL_MS = 1000 / 60; // 60 FPS
const WORKGROUP_SIZE = 8;

const canvas = document.querySelector("canvas");
if (!canvas) {
    throw new Error("No canvas found in the document.");
}

// Check for WebGPU support.
if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
}

// Request an adapter and device.
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
}

const device = await adapter.requestDevice();

// Setup the canvas for WebGPU.
; const canvasContext = canvas.getContext("webgpu");
if (!canvasContext) {
    throw new Error("No WebGPU context found.");
}

const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
canvasContext.configure({
    device: device,
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
const vertexBuffer = device.createBuffer({
    label: "Cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
        format: "float32x2" as GPUVertexFormat,
        offset: 0,
        shaderLocation: 0, // Position
    }],
};

// Create a uniform buffer that describes the grid.
const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

// Setup two storage buffers for cell states.
const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
const cellStateStorage = [
    device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
        label: "Cell State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
];

// Set each cell to a random state, then copy the JavaScript array 
// into the storage buffer.
for (let i = 0; i < cellStateArray.length; ++i) {
    cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

// Setup the shaders.
const cellShaderModule = device.createShaderModule({
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

// Create the bind group layout and pipeline layout.
const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: {} // Grid uniform buffer
    }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } // Cell state input buffer
    }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } // Cell state output buffer
    }]
});

const bindGroups = [
    device.createBindGroup({
        label: "Cell renderer bind group A",
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: uniformBuffer },
            },
            {
                binding: 1,
                resource: { buffer: cellStateStorage[0] },
            },
            {
                binding: 2,
                resource: { buffer: cellStateStorage[1] },
            },
        ],
    }),
    device.createBindGroup({
        label: "Cell renderer bind group B",
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: uniformBuffer },
            },
            {
                binding: 1,
                resource: { buffer: cellStateStorage[1] },
            },
            {
                binding: 2,
                resource: { buffer: cellStateStorage[0] },
            },
        ],
    }),
];

const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
});

const cellPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: pipelineLayout,
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

// Create the compute shader that will process the simulation.
const simulationShaderModule = device.createShaderModule({
    label: "Game of Life simulation shader",
    code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;

        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

        // Maps the cell coordinates to the index in the storage buffer,
        // wrapping within the grid (we living on a taurus baybeeeeeeeee).
        fn cellIndex(cell: vec2u) -> u32 {
          return (cell.y % u32(grid.y)) * u32(grid.x) +
            (cell.x % u32(grid.x));
        }

        fn cellActive(x: u32, y: u32) -> u32 {
          return cellStateIn[cellIndex(vec2(x, y))];
        }

        @compute
        @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
          // Determine how many active neighbors this cell has.
          let activeNeighbors =
            cellActive(cell.x + 1, cell.y + 1) +
            cellActive(cell.x + 1, cell.y    ) +
            cellActive(cell.x + 1, cell.y - 1) +
            cellActive(cell.x,     cell.y - 1) +
            cellActive(cell.x - 1, cell.y - 1) +
            cellActive(cell.x - 1, cell.y    ) +
            cellActive(cell.x - 1, cell.y + 1) +
            cellActive(cell.x,     cell.y + 1);

          let i = cellIndex(cell.xy);

          // Conway's game of life rules:
          switch activeNeighbors {
            // Active cells with 2 neighbors stay active.
            case 2: {
              cellStateOut[i] = cellStateIn[i];
            }

            // Cells with 3 neighbors become or stay active.
            case 3: {
              cellStateOut[i] = 1;
            }

            // Cells with < 2 or > 3 neighbors become inactive.
            default: {
              cellStateOut[i] = 0;
            }
          }
        }
      `
});

// Create a compute pipeline that updates the game state.
const simulationPipeline = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
    },
});

let step = 0;
function updateAndRender() {
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass({
        label: "Simulation pass",
    });

    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);

    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

    computePass.end();

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
    pass.setBindGroup(0, bindGroups[step % 2]);
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);

    // Finish and submit the render pass.
    pass.end();
    device.queue.submit([encoder.finish()]);
}

// Run update and render loop.
setInterval(updateAndRender, UPDATE_INTERVAL_MS);
