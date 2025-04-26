export const GRID_SIZE = 1024;

const WORKGROUP_SIZE = 8;

const GOL_SHADER = `
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
`;

export class LifeSimulation {
    gpu: GPU;
    device: GPUDevice;
    uniformBuffer: GPUBuffer;
    cellStateStorage: GPUBuffer[];
    bindGroupLayout: GPUBindGroupLayout;
    bindGroups: GPUBindGroup[];
    pipelineLayout: GPUPipelineLayout;
    pipeline: GPUComputePipeline;

    private constructor(
        gpu: GPU,
        device: GPUDevice,
        uniformBuffer: GPUBuffer,
        cellStateStorage: GPUBuffer[],
        bindGroupLayout: GPUBindGroupLayout,
        bindGroups: GPUBindGroup[],
        pipelineLayout: GPUPipelineLayout,
        pipeline: GPUComputePipeline,
    ) {
        this.gpu = gpu;
        this.device = device;
        this.uniformBuffer = uniformBuffer;
        this.cellStateStorage = cellStateStorage;
        this.bindGroupLayout = bindGroupLayout;
        this.bindGroups = bindGroups;
        this.pipelineLayout = pipelineLayout;
        this.pipeline = pipeline;
    }

    public static async create(gpu: GPU): Promise<LifeSimulation> {
        const adapter = await gpu.requestAdapter();
        if (!adapter) {
            throw new Error("No appropriate GPUAdapter found.");
        }

        const device = await adapter.requestDevice();
        if (!device) {
            throw new Error("No appropriate GPUDevice found.");
        }

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
            }],
        });

        const bindGroups = [
            device.createBindGroup({
                label: "Cell renderer bind group A",
                layout: bindGroupLayout,
                entries: [{
                    binding: 0,
                    resource: { buffer: uniformBuffer },
                }, {
                    binding: 1,
                    resource: { buffer: cellStateStorage[0] },
                }, {
                    binding: 2,
                    resource: { buffer: cellStateStorage[1] },
                }],
            }),
            device.createBindGroup({
                label: "Cell renderer bind group B",
                layout: bindGroupLayout,
                entries: [{
                    binding: 0,
                    resource: { buffer: uniformBuffer },
                }, {
                    binding: 1,
                    resource: { buffer: cellStateStorage[1] },
                }, {
                    binding: 2,
                    resource: { buffer: cellStateStorage[0] },
                }],
            }),
        ];

        const pipelineLayout = device.createPipelineLayout({
            label: "Cell Pipeline Layout",
            bindGroupLayouts: [bindGroupLayout],
        });

        // Create the compute shader that will process the simulation.
        const simulationShaderModule = device.createShaderModule({
            label: "Game of Life simulation shader",
            code: GOL_SHADER,
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

        return new LifeSimulation(
            gpu,
            device,
            uniformBuffer,
            cellStateStorage,
            bindGroupLayout,
            bindGroups,
            pipelineLayout,
            simulationPipeline,
        );
    }

    public buildComputePass(encoder: GPUCommandEncoder, step: number) {
        const computePass = encoder.beginComputePass({
            label: "Simulation pass",
        });

        computePass.setPipeline(this.pipeline);
        computePass.setBindGroup(0, this.bindGroups[step % 2]);

        const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

        computePass.end();
    }
}
