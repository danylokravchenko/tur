#include <metal_stdlib>
using namespace metal;

kernel void write_indices(device uint* output [[buffer(0)]],
                          constant uint& count [[buffer(1)]],
                          uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[index] = index;
    }
}
