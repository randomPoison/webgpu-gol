export const GRID_SIZE = 1024;
export const UPDATE_INTERVAL_MS = 1000 / 60;
export const WORKGROUP_SIZE = 8;

export const GOL_SHADER = `
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

export function buildComputePass(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindGroups: GPUBindGroup[], step: number) {
    const computePass = encoder.beginComputePass({
        label: "Simulation pass",
    });

    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);

    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

    computePass.end();
}
