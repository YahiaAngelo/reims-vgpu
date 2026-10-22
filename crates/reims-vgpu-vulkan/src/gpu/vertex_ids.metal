// The tests' render functions that read `[[vertex_id]]` (`encode_render.rs`), over the positions
// of `rdec_flat_v`'s buffer 0, by `vertex_id`. The fragment function writes the provoking
// vertex's ID over 255 as red, and under `bases_vertex` the draw's `[[base_vertex]]` and
// `[[base_instance]]` over 255 as green and blue. `counting_vertex` also adds one to buffer 1's
// word at its vertex ID each time it runs, a vertex function that writes memory. `point_vertex`
// and `sized_vertex` read no buffer: they place vertex `i` at the centre of pixel
// `(4 + 4k, 4 + 4k)` of a 16 by 16 target, `k` being `i % 3`, the second also writing a
// `[[point_size]]` of 1. `layer_vertex` and `viewport_vertex` write buffer 1's word at the vertex
// ID as `[[render_target_array_index]]` and `[[viewport_array_index]]`; `plain_fragment` reads no
// input and writes white, and `primitive_fragment` writes its `[[primitive_id]]`'s low byte over
// 255 as red, the next as green, and 1 as blue. `encoding_vertex` places its vertices as
// `point_vertex` does and takes an indirect command buffer through the argument buffer at buffer 1,
// whose command 0 it resets at a vertex ID past 2^24, which no test draws: a state of it can encode
// indirect commands. `corner_patch_vertex` is a post-tessellation vertex function of triangle
// patches whose three control points are float2 positions in buffer 0, its flat output its position
// in the patch, so a triangle of a patch tessellated at factor 1 takes the colour of the corner its
// provoking vertex lies at.
#include <metal_stdlib>
using namespace metal;

struct Ids {
    float4 position [[position]];
    float4 id [[flat]];
};

struct SizedIds {
    float4 position [[position]];
    float4 id [[flat]];
    float size [[point_size]];
};

constant float2 centres[3] = {float2(-0.4375, 0.4375), float2(0.0625, -0.0625),
                              float2(0.5625, -0.5625)};

vertex Ids ids_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]]) {
    return Ids{positions[vid], float4(float(vid) / 255.0, 0, 0, 1)};
}

vertex Ids bases_vertex(uint vid [[vertex_id]], uint base_vertex [[base_vertex]],
                        uint base_instance [[base_instance]],
                        const device float4 *positions [[buffer(0)]]) {
    float3 words = float3(vid, base_vertex, base_instance) / 255.0;
    return Ids{positions[vid], float4(words, 1)};
}

vertex Ids counting_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]],
                           device atomic_uint *runs [[buffer(1)]]) {
    atomic_fetch_add_explicit(&runs[vid], 1, memory_order_relaxed);
    return Ids{positions[vid], float4(float(vid) / 255.0, 0, 0, 1)};
}

vertex Ids point_vertex(uint vid [[vertex_id]]) {
    return Ids{float4(centres[vid % 3], 0, 1), float4(float(vid) / 255.0, 0, 0, 1)};
}

vertex SizedIds sized_vertex(uint vid [[vertex_id]]) {
    return SizedIds{float4(centres[vid % 3], 0, 1), float4(float(vid) / 255.0, 0, 0, 1), 1.0};
}

struct Routed {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
};

struct Viewed {
    float4 position [[position]];
    uint viewport [[viewport_array_index]];
};

vertex Routed layer_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]],
                           const device uint *layers [[buffer(1)]]) {
    return Routed{positions[vid], layers[vid]};
}

vertex Viewed viewport_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]],
                              const device uint *viewports [[buffer(1)]]) {
    return Viewed{positions[vid], viewports[vid]};
}

fragment float4 ids_fragment(Ids in [[stage_in]]) {
    return in.id;
}

fragment float4 plain_fragment() {
    return float4(1, 1, 1, 1);
}

fragment float4 primitive_fragment(uint primitive [[primitive_id]]) {
    return float4(float(primitive & 0xff) / 255, float((primitive >> 8) & 0xff) / 255, 1, 1);
}

struct Commands {
    command_buffer commands [[id(0)]];
};

vertex Ids encoding_vertex(uint vid [[vertex_id]], device Commands &held [[buffer(1)]]) {
    if (vid > 0xffffff) {
        render_command(held.commands, 0).reset();
    }
    return Ids{float4(centres[vid % 3], 0, 1), float4(float(vid) / 255.0, 0, 0, 1)};
}

[[patch(triangle, 3)]]
vertex Ids corner_patch_vertex(uint pid [[patch_id]], float3 b [[position_in_patch]],
                               const device float2 *corners [[buffer(0)]]) {
    float2 p = corners[pid * 3] * b.x + corners[pid * 3 + 1] * b.y + corners[pid * 3 + 2] * b.z;
    return Ids{float4(p, 0, 1), float4(b, 1)};
}
