// The tests' render functions for Metal's private render keys (`serialized/draws.rs`): a
// triangle over the whole target at depth `z` (vertex buffer 0), its clip distances, none, one
// or two, cutting off the target's left half and its top half, and a fragment of `colour`
// (fragment buffer 0), over the whole target or its right half. For the keys the runtime holds:
// points of a size the vertex function writes, coloured by their `[[point_coord]]`; two triangles
// of either winding, coloured by `[[front_facing]]`; a vertex function's depth comparison of a
// reference outside 0 to 1; and samples past a texture's edge, in a fragment function and in a
// kernel.
#include <metal_stdlib>
using namespace metal;

constant float2 corners[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};

struct Covered {
    float4 position [[position]];
};

struct Clipped1 {
    float4 position [[position]];
    float clip [[clip_distance]] [1];
};

struct Clipped2 {
    float4 position [[position]];
    float clip [[clip_distance]] [2];
};

vertex Covered cover(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    return Covered{float4(corners[vid], z, 1)};
}

vertex Clipped1 cover_clip1(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    Clipped1 out;
    out.position = float4(corners[vid], z, 1);
    out.clip[0] = out.position.x;
    return out;
}

vertex Clipped2 cover_clip2(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    Clipped2 out;
    out.position = float4(corners[vid], z, 1);
    out.clip[0] = out.position.x;
    out.clip[1] = -out.position.y;
    return out;
}

fragment float4 paint(constant float4 &colour [[buffer(0)]]) {
    return colour;
}

fragment float4 paint_right(float4 p [[position]], constant float4 &colour [[buffer(0)]]) {
    if (p.x < 4) {
        discard_fragment();
    }
    return colour;
}

struct Point {
    float4 position [[position]];
    float size [[point_size]];
};

// Four points of 3 pixels, one in each quarter of an 8 by 8 target, at depth `z`.
constant float2 spots[4] = {float2(-0.5, -0.5), float2(0.5, -0.5), float2(-0.5, 0.5),
                            float2(0.5, 0.5)};

vertex Point points(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    return Point{float4(spots[vid], z, 1), 3.0};
}

fragment float4 paint_coord(float2 c [[point_coord]]) {
    return float4(c, 0, 1);
}

// The target's left half counter-clockwise, its right half clockwise, at depth `z`.
constant float2 halves[6] = {float2(-1, -1), float2(0, -1), float2(-1, 1),
                             float2(1, -1), float2(0, -1), float2(1, 1)};

vertex Covered facing(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    return Covered{float4(halves[vid], z, 1)};
}

fragment float4 paint_facing(bool front [[front_facing]]) {
    return front ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);
}

struct Shaded {
    float4 position [[position]];
    float4 colour;
};

// The comparison of each of four references, -0.5, 0, 1 and 1.5, against texel (0, 0) of the
// depth texture at vertex texture 0, through the compare sampler at vertex sampler 0, in red,
// green, blue and alpha.
vertex Shaded compare_vertex(uint vid [[vertex_id]], depth2d<float> d [[texture(0)]],
                             sampler s [[sampler(0)]]) {
    const float refs[4] = {-0.5, 0.0, 1.0, 1.5};
    float4 c;
    for (int i = 0; i < 4; i++) {
        c[i] = d.sample_compare(s, float2(0.0625), refs[i], level(0));
    }
    return Shaded{float4(corners[vid], 0, 1), c};
}

fragment float4 paint_shaded(Shaded in [[stage_in]]) {
    return in.colour;
}

// Samples of the texture at fragment texture 0 through fragment sampler 0, inside it and past
// each edge, by the fragment's position.
static float4 border_sample(texture2d<float> t, sampler s, uint2 p) {
    float2 at = float2(p) * 0.5 - 1.5;
    return t.sample(s, at, level(0));
}

fragment float4 paint_border(float4 p [[position]], texture2d<float> t [[texture(0)]],
                             sampler s [[sampler(0)]]) {
    return border_sample(t, s, uint2(p.xy));
}

// `paint_border`'s samples, one a thread of an 8 by 8 grid, into buffer 0.
kernel void border_kernel(uint2 g [[thread_position_in_grid]], texture2d<float> t [[texture(0)]],
                          sampler s [[sampler(0)]], device float4 *out [[buffer(0)]]) {
    out[g.y * 8 + g.x] = border_sample(t, s, g);
}
