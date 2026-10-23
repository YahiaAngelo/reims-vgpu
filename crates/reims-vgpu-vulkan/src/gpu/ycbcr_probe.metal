// The tests' functions that read a texture of a YCbCr format (gpu/ycbcr.rs), of a two-plane XR10
// format (gpu/resource.rs's and gpu/device.rs's tests), or through a sampler (gpu/states.rs's
// tests), which build.rs compiles into ycbcr_probe.metallib: the readers write what one thread
// reads, as four halves or floats, so the tests compare Metal's texture of the format with this
// platform's bit for bit, and the rest reach a surface's bytes through a buffer over them
// (gpu/ycbcr/sampling.rs). Last, functions whose reflection the compile layer refuses
// (gpu/states.rs's tests).
#include <metal_stdlib>
using namespace metal;

// Where a probe samples: `columns` by `rows` positions from `origin`, `step` apart, in texels
// (divided by the texture's size where the sampler's coordinates are normalized), and what it
// reads there.
struct Probe {
    float2 origin;
    float2 step;
    uint columns;
    uint rows;
    // 1 where the sampler's coordinates are normalized.
    uint normalized;
    // 0 sample, 1 gather of red, 2 gather of blue.
    uint read;
};

// read() of every texel, in rows.
kernel void read_texels(texture2d<half, access::read> t [[texture(0)]],
                        device half4 *out [[buffer(0)]],
                        uint2 g [[thread_position_in_grid]]) {
    uint width = t.get_width();
    if (g.x >= width || g.y >= t.get_height()) {
        return;
    }
    out[g.y * width + g.x] = t.read(g);
}

// sample() or gather() at each of the probe's positions, in rows.
kernel void sample_texels(texture2d<half> t [[texture(0)]],
                          sampler s [[sampler(0)]],
                          device half4 *out [[buffer(0)]],
                          constant Probe &p [[buffer(1)]],
                          uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    half4 v;
    if (p.read == 1) {
        v = t.gather(s, at, int2(0), component::x);
    } else if (p.read == 2) {
        v = t.gather(s, at, int2(0), component::z);
    } else {
        v = t.sample(s, at);
    }
    out[g.y * p.columns + g.x] = v;
}

// read() of every texel, in rows, as floats: the texture's values at full precision.
kernel void read_texels_float(texture2d<float, access::read> t [[texture(0)]],
                              device float4 *out [[buffer(0)]],
                              uint2 g [[thread_position_in_grid]]) {
    uint width = t.get_width();
    if (g.x >= width || g.y >= t.get_height()) {
        return;
    }
    out[g.y * width + g.x] = t.read(g);
}

// sample() or gather() at each of the probe's positions, in rows, as floats.
kernel void sample_texels_float(texture2d<float> t [[texture(0)]],
                                sampler s [[sampler(0)]],
                                device float4 *out [[buffer(0)]],
                                constant Probe &p [[buffer(1)]],
                                uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    float4 v;
    if (p.read == 1) {
        v = t.gather(s, at, int2(0), component::x);
    } else if (p.read == 2) {
        v = t.gather(s, at, int2(0), component::z);
    } else {
        v = t.sample(s, at);
    }
    out[g.y * p.columns + g.x] = v;
}

// read_texels_float of slice 0 of a 2D array: a view of a texture as a 2D array.
kernel void read_slice_texels_float(texture2d_array<float, access::read> t [[texture(0)]],
                                    device float4 *out [[buffer(0)]],
                                    uint2 g [[thread_position_in_grid]]) {
    uint width = t.get_width();
    if (g.x >= width || g.y >= t.get_height()) {
        return;
    }
    out[g.y * width + g.x] = t.read(g, 0);
}

// sample_texels_float of slice 0 of a 2D array.
kernel void sample_slice_texels_float(texture2d_array<float> t [[texture(0)]],
                                      sampler s [[sampler(0)]],
                                      device float4 *out [[buffer(0)]],
                                      constant Probe &p [[buffer(1)]],
                                      uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    float4 v;
    if (p.read == 1) {
        v = t.gather(s, at, 0, int2(0), component::x);
    } else if (p.read == 2) {
        v = t.gather(s, at, 0, int2(0), component::z);
    } else {
        v = t.sample(s, at, 0);
    }
    out[g.y * p.columns + g.x] = v;
}

// sample() at each of the probe's positions, in rows, whatever its read: a reader with no gather,
// for a texture of a format the runtime gathers none of (the two-plane XR10 formats,
// gpu/resource.rs's tests), as a function holding a gather it refuses.
kernel void sample_only_texels(texture2d<half> t [[texture(0)]],
                               sampler s [[sampler(0)]],
                               device half4 *out [[buffer(0)]],
                               constant Probe &p [[buffer(1)]],
                               uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    out[g.y * p.columns + g.x] = t.sample(s, at);
}

// sample_only_texels into floats: also the reader through a cubic sampler under mirrored repeat,
// which the runtime refuses in a function holding a gather (gpu/states.rs's tests).
kernel void sample_only_texels_float(texture2d<float> t [[texture(0)]],
                                     sampler s [[sampler(0)]],
                                     device float4 *out [[buffer(0)]],
                                     constant Probe &p [[buffer(1)]],
                                     uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    out[g.y * p.columns + g.x] = t.sample(s, at);
}

// sample() of texture 0 at each texel's centre through a constexpr sampler, nearest and
// repeating, in rows.
kernel void sample_constexpr(texture2d<half> t [[texture(0)]],
                             device half4 *out [[buffer(0)]],
                             uint2 g [[thread_position_in_grid]]) {
    constexpr sampler s(address::repeat, filter::nearest);
    uint width = t.get_width();
    if (g.x >= width || g.y >= t.get_height()) {
        return;
    }
    out[g.y * width + g.x] = t.sample(s, (float2(g) + 0.5f) / float2(width, t.get_height()));
}

// sample() at each of the probe's positions, in rows, normalized, through a constexpr sampler,
// nearest and clamped to the edge, which an address-mode map of every mode to clamp_to_edge
// keeps as it is.
kernel void sample_constexpr_clamped(texture2d<half> t [[texture(0)]],
                                     device half4 *out [[buffer(0)]],
                                     constant Probe &p [[buffer(1)]],
                                     uint2 g [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_edge, filter::nearest);
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 at = (p.origin + float2(g) * p.step) / float2(t.get_width(), t.get_height());
    out[g.y * p.columns + g.x] = t.sample(s, at);
}

// sample() at each of the probe's positions, in rows, normalized, through a constexpr sampler,
// linear and clamped to the edge, into halves.
kernel void sample_constexpr_linear(texture2d<half> t [[texture(0)]],
                                    device half4 *out [[buffer(0)]],
                                    constant Probe &p [[buffer(1)]],
                                    uint2 g [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 at = (p.origin + float2(g) * p.step) / float2(t.get_width(), t.get_height());
    out[g.y * p.columns + g.x] = t.sample(s, at);
}

// sample_constexpr_linear into floats.
kernel void sample_constexpr_linear_float(texture2d<float> t [[texture(0)]],
                                          device float4 *out [[buffer(0)]],
                                          constant Probe &p [[buffer(1)]],
                                          uint2 g [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 at = (p.origin + float2(g) * p.step) / float2(t.get_width(), t.get_height());
    out[g.y * p.columns + g.x] = t.sample(s, at);
}

// sample() at each of the probe's positions, in rows, normalized, through a constexpr sampler of
// `mode` on both axes and `how` filtering, into floats: the modes a texture's address-mode map
// may change.
#define SAMPLE_CONSTEXPR(name, mode, how)                                                         \
    kernel void name(texture2d<float> t [[texture(0)]],                                           \
                     device float4 *out [[buffer(0)]],                                            \
                     constant Probe &p [[buffer(1)]],                                             \
                     uint2 g [[thread_position_in_grid]]) {                                       \
        constexpr sampler s(address::mode, filter::how);                                          \
        if (g.x >= p.columns || g.y >= p.rows) {                                                  \
            return;                                                                               \
        }                                                                                         \
        float2 at = (p.origin + float2(g) * p.step) / float2(t.get_width(), t.get_height());      \
        out[g.y * p.columns + g.x] = t.sample(s, at);                                             \
    }
SAMPLE_CONSTEXPR(sample_constexpr_repeat, repeat, nearest)
SAMPLE_CONSTEXPR(sample_constexpr_mirrored_repeat, mirrored_repeat, nearest)
SAMPLE_CONSTEXPR(sample_constexpr_zero, clamp_to_zero, nearest)
SAMPLE_CONSTEXPR(sample_constexpr_repeat_linear, repeat, linear)
SAMPLE_CONSTEXPR(sample_constexpr_mirrored_repeat_linear, mirrored_repeat, linear)
SAMPLE_CONSTEXPR(sample_constexpr_bicubic, clamp_to_edge, bicubic)
SAMPLE_CONSTEXPR(sample_constexpr_repeat_bicubic, repeat, bicubic)

// SAMPLE_CONSTEXPR under clamp to the border colour, `colour` and `how` filtering.
#define SAMPLE_CONSTEXPR_BORDER(name, colour, how)                                                \
    kernel void name(texture2d<float> t [[texture(0)]],                                           \
                     device float4 *out [[buffer(0)]],                                            \
                     constant Probe &p [[buffer(1)]],                                             \
                     uint2 g [[thread_position_in_grid]]) {                                       \
        constexpr sampler s(address::clamp_to_border, border_color::colour, filter::how);         \
        if (g.x >= p.columns || g.y >= p.rows) {                                                  \
            return;                                                                               \
        }                                                                                         \
        float2 at = (p.origin + float2(g) * p.step) / float2(t.get_width(), t.get_height());      \
        out[g.y * p.columns + g.x] = t.sample(s, at);                                             \
    }
SAMPLE_CONSTEXPR_BORDER(sample_constexpr_border_white, opaque_white, nearest)
SAMPLE_CONSTEXPR_BORDER(sample_constexpr_border_black_linear, opaque_black, linear)
SAMPLE_CONSTEXPR_BORDER(sample_constexpr_border_clear_linear, transparent_black, linear)

// SAMPLE_CONSTEXPR in a fragment function, of the sampler `...` states: sample() at the probe's
// position of each fragment of a target of the probe's columns by rows, normalized, into floats,
// in rows, as sample_fragment_float does through sampler 0.
#define SAMPLE_CONSTEXPR_FRAGMENT(name, ...)                                                      \
    fragment float4 name(float4 position [[position]],                                           \
                         texture2d<float> t [[texture(0)]],                                       \
                         device float4 *out [[buffer(0)]],                                        \
                         constant Probe &p [[buffer(1)]]) {                                       \
        constexpr sampler s(__VA_ARGS__);                                                         \
        uint2 g = uint2(position.xy);                                                             \
        float2 at = (p.origin + float2(g) * p.step) / float2(t.get_width(), t.get_height());      \
        out[g.y * p.columns + g.x] = t.sample(s, at);                                             \
        return float4(0, 0, 0, 1);                                                                \
    }
SAMPLE_CONSTEXPR_FRAGMENT(sample_constexpr_repeat_linear_fragment, address::repeat, filter::linear)
SAMPLE_CONSTEXPR_FRAGMENT(sample_constexpr_mirrored_repeat_fragment, address::mirrored_repeat,
                          filter::nearest)
SAMPLE_CONSTEXPR_FRAGMENT(sample_constexpr_border_clear_linear_fragment, address::clamp_to_border,
                          border_color::transparent_black, filter::linear)

// What an argument buffer holds of a sampler: its resource ID, which the CPU writes.
struct HeldSampler {
    sampler s [[id(0)]];
};

// sample() of texture 0 at each of the probe's positions, in rows, normalized, through the sampler
// an argument buffer holds, into floats.
kernel void sample_held_sampler_float(texture2d<float> t [[texture(0)]],
                                      device float4 *out [[buffer(0)]],
                                      constant Probe &p [[buffer(1)]],
                                      constant HeldSampler &held [[buffer(2)]],
                                      uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 at = (p.origin + float2(g) * p.step) / float2(t.get_width(), t.get_height());
    out[g.y * p.columns + g.x] = t.sample(held.s, at);
}

// What an argument buffer holds of a texture and the sampler it is sampled through: their
// resource IDs, which the CPU writes.
struct HeldSampled {
    texture2d<float> t [[id(0)]];
    sampler s [[id(1)]];
};

// sample() of the texture an argument buffer holds at each of the probe's positions, in rows,
// normalized, through the sampler it holds beside it, into floats.
kernel void sample_held_float(device float4 *out [[buffer(0)]],
                              constant Probe &p [[buffer(1)]],
                              constant HeldSampled &held [[buffer(2)]],
                              uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.columns || g.y >= p.rows) {
        return;
    }
    float2 size = float2(held.t.get_width(), held.t.get_height());
    float2 at = (p.origin + float2(g) * p.step) / size;
    out[g.y * p.columns + g.x] = held.t.sample(held.s, at);
}

// sample() of the texture an argument buffer holds at each fragment's position, normalized by
// the texture's size, through the sampler it holds beside it: a fragment stage that samples
// through an argument buffer's sampler.
fragment float4 sample_held_fragment(float4 position [[position]],
                                     constant HeldSampled &held [[buffer(0)]]) {
    float2 size = float2(held.t.get_width(), held.t.get_height());
    return held.t.sample(held.s, position.xy / size);
}

// The sum of textures 0 and 1 sampled through sampler 0 at each texel's centre of texture 0, in
// rows: one sampler slot that samples two textures.
kernel void sample_two(texture2d<half> a [[texture(0)]],
                       texture2d<half> b [[texture(1)]],
                       sampler s [[sampler(0)]],
                       device half4 *out [[buffer(0)]],
                       uint2 g [[thread_position_in_grid]]) {
    uint width = a.get_width();
    if (g.x >= width || g.y >= a.get_height()) {
        return;
    }
    float2 at = (float2(g) + 0.5f) / float2(width, a.get_height());
    out[g.y * width + g.x] = a.sample(s, at) + b.sample(s, at);
}

// What an argument buffer holds of a texture: its resource ID, which the CPU writes.
struct Held {
    texture2d<half, access::read> t [[id(0)]];
};

// read() of every texel of the texture an argument buffer holds, in rows.
kernel void read_held(device half4 *out [[buffer(0)]],
                      constant Held &held [[buffer(1)]],
                      uint2 g [[thread_position_in_grid]]) {
    uint width = held.t.get_width();
    if (g.x >= width || g.y >= held.t.get_height()) {
        return;
    }
    out[g.y * width + g.x] = held.t.read(g);
}

// read() of texture 0 into texture 1, texel by texel: a kernel that writes a texture other than
// the one it reads.
kernel void copy_texels(texture2d<half, access::read> from [[texture(0)]],
                        texture2d<half, access::write> to [[texture(1)]],
                        uint2 g [[thread_position_in_grid]]) {
    if (g.x >= from.get_width() || g.y >= from.get_height()) {
        return;
    }
    to.write(from.read(g), g);
}

// Writes 0 into texture 0's texel at each thread.
kernel void write_texels(texture2d<half, access::write> t [[texture(0)]],
                         uint2 g [[thread_position_in_grid]]) {
    t.write(half4(0.0h), g);
}

// write_texels in a fragment function: writes 0 into texture 0's texel at each fragment, which
// a draw over cover_vertex into a target of the texture's size reaches once each.
fragment float4 write_texels_fragment(float4 position [[position]],
                                      texture2d<half, access::write> t [[texture(0)]]) {
    t.write(half4(0.0h), uint2(position.xy));
    return float4(0.0);
}

// Writes 0 into buffer 0's 32-bit word at each thread: a write of a surface's bytes through a
// buffer over them.
kernel void zero_words(device uint *b [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    b[i] = 0;
}

// Copies buffer 0's 32-bit word at each thread into buffer 1: a read of a surface's bytes through
// a buffer over them, which never writes them.
kernel void copy_words(device const uint *from [[buffer(0)]], device uint *to [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    to[i] = from[i];
}

// A triangle over the whole target.
vertex float4 cover_vertex(uint v [[vertex_id]]) {
    const float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(p[v], 0, 1);
}

// A triangle over the target's top left half: what a pass draws shows how its texels land.
vertex float4 corner_vertex(uint v [[vertex_id]]) {
    const float2 p[3] = {float2(-1, 1), float2(1, 1), float2(-1, -1)};
    return float4(p[v], 0, 1);
}

// A colour of each fragment's position in a 32 by 32 frame: what a pass stores shows where each
// of its texels lands.
fragment float4 gradient_fragment(float4 position [[position]]) {
    return float4(position.x / 32.0, position.y / 32.0, 0.25, 1);
}

// sample() at the probe's position of each fragment of a target of the probe's columns by rows,
// through sampler 0, into halves, in rows: sample_texels' sample in a fragment function.
fragment float4 sample_fragment(float4 position [[position]],
                                texture2d<half> t [[texture(0)]],
                                sampler s [[sampler(0)]],
                                device half4 *out [[buffer(0)]],
                                constant Probe &p [[buffer(1)]]) {
    uint2 g = uint2(position.xy);
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    out[g.y * p.columns + g.x] = t.sample(s, at);
    return float4(0, 0, 0, 1);
}

// sample_fragment into floats.
fragment float4 sample_fragment_float(float4 position [[position]],
                                      texture2d<float> t [[texture(0)]],
                                      sampler s [[sampler(0)]],
                                      device float4 *out [[buffer(0)]],
                                      constant Probe &p [[buffer(1)]]) {
    uint2 g = uint2(position.xy);
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    out[g.y * p.columns + g.x] = t.sample(s, at);
    return float4(0, 0, 0, 1);
}

// sample_texels in a fragment function: sample() or gather() at the probe's position of each
// fragment of a target of the probe's columns by rows, through sampler 0, into halves, in rows.
fragment float4 probe_fragment(float4 position [[position]],
                               texture2d<half> t [[texture(0)]],
                               sampler s [[sampler(0)]],
                               device half4 *out [[buffer(0)]],
                               constant Probe &p [[buffer(1)]]) {
    uint2 g = uint2(position.xy);
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    half4 v;
    if (p.read == 1) {
        v = t.gather(s, at, int2(0), component::x);
    } else if (p.read == 2) {
        v = t.gather(s, at, int2(0), component::z);
    } else {
        v = t.sample(s, at);
    }
    out[g.y * p.columns + g.x] = v;
    return float4(0, 0, 0, 1);
}

// probe_fragment into floats: sample_texels_float in a fragment function.
fragment float4 probe_fragment_float(float4 position [[position]],
                                     texture2d<float> t [[texture(0)]],
                                     sampler s [[sampler(0)]],
                                     device float4 *out [[buffer(0)]],
                                     constant Probe &p [[buffer(1)]]) {
    uint2 g = uint2(position.xy);
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    float4 v;
    if (p.read == 1) {
        v = t.gather(s, at, int2(0), component::x);
    } else if (p.read == 2) {
        v = t.gather(s, at, int2(0), component::z);
    } else {
        v = t.sample(s, at);
    }
    out[g.y * p.columns + g.x] = v;
    return float4(0, 0, 0, 1);
}

// An argument buffer holding an array of structs that hold a struct and no constant, whose
// stride in a pipeline's reflection Metal's answer is unmeasured for: the compile layer refuses
// the reflection of a function that takes one (gpu/states.rs's tests).
struct HeldPointer {
    device float *p;
};
struct HoldingElement {
    HeldPointer held;
};
struct NestedArguments {
    HoldingElement e[2];
};

// Writes 1 through the pointer of element `i & 1` of an argument buffer of NestedArguments.
kernel void nested_arguments_kernel(device NestedArguments &a [[buffer(0)]],
                                    uint i [[thread_position_in_grid]]) {
    *a.e[i & 1].held.p = 1.0f;
}

// cover_vertex, writing 1 through the pointer of element `v & 1` of an argument buffer of
// NestedArguments.
vertex float4 nested_arguments_vertex(uint v [[vertex_id]],
                                      device NestedArguments &a [[buffer(0)]]) {
    *a.e[v & 1].held.p = 1.0f;
    const float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(p[v], 0, 1);
}
