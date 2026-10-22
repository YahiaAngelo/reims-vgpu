//! What the private fills write, and where: `fillBuffer:range:pattern4:`,
//! `fillTexture:level:slice:region:bytes:length:` and
//! `fillTexture:level:slice:region:color:pixelFormat:`. The blit encoder writes the texture
//! fills' bytes with the runtime's copies, and into a texture of a format whose fill nothing here
//! measured ([`texel_bytes`]) with the runtime's fill ([`crate::gpu::encode_blit`]). A colour in a
//! bfloat format, an extended-range `_sRGB` twin or an RGBX format ([`RUNTIME_PACKED`]) is the
//! runtime's colour fill, which packs it. The pattern fill is the runtime's
//! (`fill_buffer_pattern4`), which writes it as Metal does (below).
//!
//! Metal's answers, measured on `AppleM3Max-40c-25G229` with no validation:
//! - **The pattern** of `pattern4:` is its four bytes, low first, repeated from the range's start
//!   up to the first offset that is a multiple of 16 (the head), and from that offset afresh to
//!   the range's end: a fill of 7 bytes at 41 writes `01 02 03 04 01 02 03`, then `01 02 ..` from
//!   48. An empty range writes nothing (`encode_blit`'s `pattern_fills_are_the_metal_platforms`).
//! - **A texel** of either texture fill is its pattern (the bytes, or the colour's bytes) repeated
//!   from its start and cut at the texture format's block bytes ([`texel`]): bytes past a block
//!   go unread, and a colour of fewer bytes repeats (an R8Unorm colour fills each byte of an
//!   RGBA32Uint texel).
//! - **A colour** is packed in the call's pixel format, whatever the texture's ([`color_bytes`]).
//!   An unsigned normalized component is the colour's as an `f32`, clamped to [0, 1], times its
//!   largest value (exactly for 4 and 8 bits, as an `f32` for 1, 2, 5, 6, 10 and 16), rounded to
//!   nearest with halves away from zero; a signed normalized one likewise in [-1, 1], as an
//!   `f32`; an integer one the colour's `f64` clamped to the type and truncated toward zero, but
//!   a 32-bit unsigned one strictly between 2^31 - 1 and 2^31 is 2^31; a half float the `f32`
//!   rounded to nearest even, overflowing to infinity. A NaN is 0 in a normalized component. In
//!   an integer one a quiet NaN (the `f64`'s top mantissa bit set) is 0, or the least value in a
//!   signed one, and a signalling NaN the largest value. In a float one it keeps its sign and the
//!   top bits of its `f64`'s mantissa, as many as the float's mantissa holds, the quiet bit set.
//!   Measured: 36 NaNs, quiet and signalling, of either sign, whose payloads set the bits a half
//!   and a float keep and those below them, in every format packed here, and doubles on and
//!   beside the 8-, 16- and 32-bit integer edges
//!   (`nan_payloads_and_integer_edges_are_the_metal_platforms_here`).
//! - **ARGB8Unorm** is RGBA8Unorm's components rotated: alpha, red, green, blue, each as
//!   RGBA8Unorm's. Measured: 4506 colours of each value of the tests' sweep (zeroes, ones, NaNs,
//!   infinities, denormals, the halves between every 8-bit step, values past [-1, 1]) in each
//!   component; RGBA8Unorm's own order misses 3208 of them.
//! - **An sRGB component** (of R8Unorm_sRGB, RG8Unorm_sRGB, RGBA8Unorm_sRGB and
//!   BGRA8Unorm_sRGB; alpha stays unsigned normalized) is the code whose decoded value is nearest
//!   the colour's `f32`, a value halfway between two taking the greater ([`srgb`]): not the
//!   standard encoding curve's value rounded, which differs near some thresholds. Measured: every
//!   one of the 255 thresholds, the least `f64` colour that stores each code, of RGBA8Unorm_sRGB's
//!   red, green and blue, BGRA8Unorm_sRGB's red, R8Unorm_sRGB's and RG8Unorm_sRGB's green; each
//!   lies halfway between two `f32`s, as the colour is rounded to an `f32` first.
//! - **An extended-range component** (BGRA10_XR's blue, green and red, BGR10_XR's) is the
//!   colour's `f32` plus `384 / 510`, times 510, each rounded to an `f32`, plus a half, rounded
//!   down and clamped to the code's 10 bits ([`extended`]): the code of value `(code - 384) /
//!   510` nearest it, as Metal's own `f32` arithmetic finds it. Measured: every one of the 1023
//!   thresholds of BGRA10_XR's blue. BGRA10_XR's alpha is the colour's clamped to [0, 1] first,
//!   a NaN as 0; BGR10_XR keeps no alpha and its top two bits 0; each of BGRA10_XR's components
//!   lies in the top 10 bits of its 16, below them 0.
//! - **RG11B10Float's components** are unsigned floats of a 5-bit exponent and 6 bits of mantissa
//!   (red, green) or 5 (blue), from the colour's `f32` ([`small_float`]): rounded to nearest even,
//!   but below the least normal, 2^-14, the bits shifted past the mantissa's are dropped first,
//!   so a value a hair above a halfway point rounds as the halfway point; past the largest finite
//!   value the largest, an infinity the infinity, a NaN of either sign all ones, a negative value
//!   0. Alpha is dropped.
//! - **RGB9E5Float** is red, green and blue, from their `f32`s, each clamped to [0, 65408], a NaN
//!   0, with a 5-bit exponent shared by all three ([`shared_exponent`]): the largest's, at least
//!   2^-16, after its top 9 bits are rounded to nearest, halves up; each mantissa is its
//!   component over that exponent's step, rounded to nearest, halves up.
//! - Measured for both: every value of a sweep of 4850 colours (randomly drawn values, zeroes,
//!   infinities, NaNs, the range's ends) and of 6406 (RG11B10Float) and 5932 (RGB9E5Float) more on
//!   and beside every rounding halfway point of each exponent, denormals included, as the values'
//!   own `f64`s, `f32` steps and smaller steps an `f32` absorbs.
//! - BGRA10_XR_sRGB and BGR10_XR_sRGB are not packed here. Metal's thresholds (measured: all
//!   1023 of BGRA10_XR_sRGB's blue, the same in BGR10_XR_sRGB's green) are not the standard
//!   curve's encoding rounded, and lie near the midpoints of the codes' decoded values, as the
//!   8-bit sRGB codes' do; but no `f32` or `f64` reproduction of the decoding found here lands on
//!   them: the nearest misses 237 thresholds of the curve's power segment by an `f32` step, and
//!   up to 142 steps on its linear segment, whose thresholds are not symmetric about 0. The
//!   runtime packs them by a table of the thresholds measured on an Apple M4, and
//!   bfloat by rounding to nearest even, a NaN keeping its sign and payload's top bits; a colour
//!   fill in one of these formats is the runtime's ([`RUNTIME_PACKED`]), whose bytes
//!   `encode_blit`'s `colours_the_runtime_packs_are_the_metal_platforms` compares with Metal's
//!   on this host. So is one in an RGBX format, which the runtime packs as its RGBA twin's red,
//!   green and blue, the X field zeros where the twin's alpha is normalized or an integer and
//!   the alpha's conversion where it is a float.
//! - **A colour in a format Metal packs none in** (Invalid, every depth and stencil format:
//!   Depth16Unorm, Depth32Float, Stencil8, Depth24Unorm_Stencil8, Depth32Float_Stencil8,
//!   X32_Stencil8, X24_Stencil8, and the two-plane XR10 formats RGB10A8_2P_XR10 and its sRGB
//!   twin, 550 and 551: [`LEFTOVER_FORMATS`]) writes no byte of the colour, but bytes Metal holds
//!   from earlier work (measured): after an RGBA32Uint fill of (7, 9, 3, 1) into one texture, a
//!   fill of any colour in one of these formats later in the encoder writes `07 00 00 00 09 00
//!   00 00 03 00 00 00 01 00 00 00` into another texture's RGBA32Uint texel; with no colour
//!   packed before it in the encoder, bytes no colour gave (not a colour an earlier command
//!   buffer packed), which differ between these formats, between texture formats and between
//!   runs (a 550 colour into a 550 texture, paged out by core's XR10 kernel: plane 0 `77 d7 44
//!   c7` in one process and `00 00 00 c0` in another, whatever the colour, NaN included). They
//!   are Metal's state, which nothing here holds, so these die too
//!   (`metal_fills_a_colour_it_packs_none_in_with_an_earlier_fills_bytes`).
//! - **Where** ([`place`]): a level past the texture's writes nothing; the region is cut at the
//!   level's extent. A 3D texture takes the region's depth and ignores the slice; every other
//!   type ignores the region's `z` and depth. A 1D or 2D texture ignores the slice; an array
//!   writes a slice it has and nothing past them; a cube writes face `slice % 6`, a cube array
//!   slice `slice` while it has it. Metal's work grows with the region before the cut (2^32
//!   texels: 23 ms; 2^64 did not end in 10 minutes), so a region past 2^32 texels dies, as
//!   nothing measured how it ends.

use metal2vulkan::metal::{PixelFormat, TextureType};
use reims_vgpu_core::platform::types::{ClearColor, MtlRegion};

use crate::answer::unrepresentable;

/// The most texels a fill region may span before the cut: Metal took 23 ms for this many.
const MEASURED_TEXELS: u64 = 1 << 32;

/// The most bytes of its pattern a fill of a texture with bytes reads: a block of the largest
/// format Metal describes at any level the runtime measured (`RGBA32Float` and the 16-byte
/// compressed blocks), as a texel is the pattern cut at its format's block ([`texel`], and the
/// runtime's fill likewise), the bytes past it unread.
pub const MOST_PATTERN_BYTES: usize = 16;

/// A texel of `bytes` bytes: `pattern` repeated from its start and cut at `bytes`. An empty
/// pattern (none reaches here) is a texel of zeroes.
pub fn texel(pattern: &[u8], bytes: usize) -> Vec<u8> {
    if pattern.is_empty() {
        return vec![0; bytes];
    }
    pattern.iter().copied().cycle().take(bytes).collect()
}

/// The bytes of a texel of `format`, a texture's, that a fill writes as this module measured it;
/// `None` for a format whose fill nothing here measured, which the runtime's fill writes
/// ([`crate::gpu::encode_blit`]): a compressed one (the 4:2:2 `GBGR422` and `BGRG422` among
/// them), a stencil or combined depth/stencil one, the two-plane XR10 ones (550, 551), whose
/// colour word is not the block Metal describes, and one Metal describes no block of. `sel` is
/// the fill's, which dies at a format another release describes and macOS 26.7, where the fills
/// were measured, does not ([`crate::gpu::format_info::block`]).
pub fn texel_bytes(sel: &'static str, format: PixelFormat) -> Option<usize> {
    let raw = u64::from(format.as_raw());
    let block = crate::gpu::format_info::block(sel, raw, "a fill of a texture of");
    let unmeasured = block.compressed()
        || block.bytes() == 0
        || matches!(
            format,
            PixelFormat::Stencil8
                | PixelFormat::X32Stencil8
                | PixelFormat::X24Stencil8
                | PixelFormat::Depth24UnormStencil8
                | PixelFormat::Depth32FloatStencil8
                | PixelFormat::RGB10A82PXR10
                | PixelFormat::RGB10A82PXR10SRGB
        );
    match unmeasured {
        true => None,
        false => usize::try_from(block.bytes()).ok(),
    }
}

/// How a component is encoded.
#[derive(Debug, Clone, Copy)]
enum Code {
    /// Unsigned normalized, of this many bits.
    Unorm(u32),
    /// Signed normalized, of this many bits.
    Snorm(u32),
    /// Unsigned integer, of this many bits.
    Uint(u32),
    /// Signed integer, of this many bits.
    Sint(u32),
    /// A half float.
    Half,
    /// A float.
    Float,
    /// An 8-bit sRGB component ([`srgb`]).
    Srgb,
    /// A 10-bit extended-range component ([`extended`]).
    Extended,
    /// A 10-bit extended-range alpha: the colour's clamped to [0, 1] first, a NaN 0.
    ExtendedAlpha,
    /// An unsigned float of a 5-bit exponent and this many bits of mantissa ([`small_float`]).
    SmallFloat(u32),
}

/// A component of a packed colour: which of red, green, blue and alpha (0 to 3), its code and
/// its first bit.
type Field = (usize, Code, u32);

/// The components of `format`'s colour, and its bytes; `None` for a format this platform packs
/// no colour in.
fn layout(format: PixelFormat) -> Option<(Vec<Field>, usize)> {
    use Code::{Extended, ExtendedAlpha, Float, Half, Sint, SmallFloat, Snorm, Srgb, Uint, Unorm};
    use PixelFormat as F;
    /// The components red, green, blue and alpha, as many as `codes`, each `bits` wide.
    fn channels(codes: &[Code], bits: u32) -> (Vec<Field>, usize) {
        let fields = (0u32..)
            .zip(codes)
            .map(|(i, &code)| (i as usize, code, i * bits))
            .collect();
        (fields, codes.len() * (bits as usize / 8))
    }
    /// The colour formats of one width: `n` components of `bits` each.
    fn plain(code: Code, n: usize, bits: u32) -> (Vec<Field>, usize) {
        channels(&[code; 4][..n], bits)
    }
    Some(match format {
        F::A8Unorm => (vec![(3, Unorm(8), 0)], 1),
        F::R8Unorm => plain(Unorm(8), 1, 8),
        F::R8UnormSRGB => plain(Srgb, 1, 8),
        F::R8Snorm => plain(Snorm(8), 1, 8),
        F::R8Uint => plain(Uint(8), 1, 8),
        F::R8Sint => plain(Sint(8), 1, 8),
        F::R16Unorm => plain(Unorm(16), 1, 16),
        F::R16Snorm => plain(Snorm(16), 1, 16),
        F::R16Uint => plain(Uint(16), 1, 16),
        F::R16Sint => plain(Sint(16), 1, 16),
        F::R16Float => plain(Half, 1, 16),
        F::RG8Unorm => plain(Unorm(8), 2, 8),
        F::RG8UnormSRGB => plain(Srgb, 2, 8),
        F::RG8Snorm => plain(Snorm(8), 2, 8),
        F::RG8Uint => plain(Uint(8), 2, 8),
        F::RG8Sint => plain(Sint(8), 2, 8),
        F::B5G6R5Unorm => (
            vec![(2, Unorm(5), 0), (1, Unorm(6), 5), (0, Unorm(5), 11)],
            2,
        ),
        F::A1BGR5Unorm => (
            vec![
                (3, Unorm(1), 0),
                (2, Unorm(5), 1),
                (1, Unorm(5), 6),
                (0, Unorm(5), 11),
            ],
            2,
        ),
        F::ABGR4Unorm => (
            vec![
                (3, Unorm(4), 0),
                (2, Unorm(4), 4),
                (1, Unorm(4), 8),
                (0, Unorm(4), 12),
            ],
            2,
        ),
        F::BGR5A1Unorm => (
            vec![
                (2, Unorm(5), 0),
                (1, Unorm(5), 5),
                (0, Unorm(5), 10),
                (3, Unorm(1), 15),
            ],
            2,
        ),
        F::R32Uint => plain(Uint(32), 1, 32),
        F::R32Sint => plain(Sint(32), 1, 32),
        F::R32Float => plain(Float, 1, 32),
        F::RG16Unorm => plain(Unorm(16), 2, 16),
        F::RG16Snorm => plain(Snorm(16), 2, 16),
        F::RG16Uint => plain(Uint(16), 2, 16),
        F::RG16Sint => plain(Sint(16), 2, 16),
        F::RG16Float => plain(Half, 2, 16),
        F::RGBA8Unorm => plain(Unorm(8), 4, 8),
        F::RGBA8UnormSRGB => (
            vec![(0, Srgb, 0), (1, Srgb, 8), (2, Srgb, 16), (3, Unorm(8), 24)],
            4,
        ),
        F::RGBA8Snorm => plain(Snorm(8), 4, 8),
        F::RGBA8Uint => plain(Uint(8), 4, 8),
        F::RGBA8Sint => plain(Sint(8), 4, 8),
        F::ARGB8Unorm => (
            vec![
                (3, Unorm(8), 0),
                (0, Unorm(8), 8),
                (1, Unorm(8), 16),
                (2, Unorm(8), 24),
            ],
            4,
        ),
        F::BGRA8Unorm => (
            vec![
                (2, Unorm(8), 0),
                (1, Unorm(8), 8),
                (0, Unorm(8), 16),
                (3, Unorm(8), 24),
            ],
            4,
        ),
        F::BGRA8UnormSRGB => (
            vec![(2, Srgb, 0), (1, Srgb, 8), (0, Srgb, 16), (3, Unorm(8), 24)],
            4,
        ),
        F::BGRA10XR => (
            vec![
                (2, Extended, 6),
                (1, Extended, 22),
                (0, Extended, 38),
                (3, ExtendedAlpha, 54),
            ],
            8,
        ),
        F::BGR10XR => (
            vec![(2, Extended, 0), (1, Extended, 10), (0, Extended, 20)],
            4,
        ),
        F::RG11B10Float => (
            vec![
                (0, SmallFloat(6), 0),
                (1, SmallFloat(6), 11),
                (2, SmallFloat(5), 22),
            ],
            4,
        ),
        F::RGB10A2Unorm => (
            vec![
                (0, Unorm(10), 0),
                (1, Unorm(10), 10),
                (2, Unorm(10), 20),
                (3, Unorm(2), 30),
            ],
            4,
        ),
        F::RGB10A2Uint => (
            vec![
                (0, Uint(10), 0),
                (1, Uint(10), 10),
                (2, Uint(10), 20),
                (3, Uint(2), 30),
            ],
            4,
        ),
        F::BGR10A2Unorm => (
            vec![
                (2, Unorm(10), 0),
                (1, Unorm(10), 10),
                (0, Unorm(10), 20),
                (3, Unorm(2), 30),
            ],
            4,
        ),
        F::RG32Uint => plain(Uint(32), 2, 32),
        F::RG32Sint => plain(Sint(32), 2, 32),
        F::RG32Float => plain(Float, 2, 32),
        F::RGBA16Unorm => plain(Unorm(16), 4, 16),
        F::RGBA16Snorm => plain(Snorm(16), 4, 16),
        F::RGBA16Uint => plain(Uint(16), 4, 16),
        F::RGBA16Sint => plain(Sint(16), 4, 16),
        F::RGBA16Float => plain(Half, 4, 16),
        F::RGBA32Uint => plain(Uint(32), 4, 32),
        F::RGBA32Sint => plain(Sint(32), 4, 32),
        F::RGBA32Float => plain(Float, 4, 32),
        _ => return None,
    })
}

/// `bits` ones.
fn ones(bits: u32) -> u64 {
    (1u64 << bits) - 1
}

/// `v`, clamped to [`lo`, `hi`] and scaled by `scale`, rounded to nearest (halves away from
/// zero); a NaN is 0. The product is exact where `exact`, and an `f32` otherwise, as Metal's
/// is for a 4- or 8-bit unsigned normalized component and for every other (measured).
fn normalized(v: f32, lo: f32, hi: f32, scale: u64, exact: bool) -> i64 {
    if v.is_nan() {
        return 0;
    }
    let v = v.clamp(lo, hi);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "in range of an i64 by the clamp; the scales are at most 2^16 - 1, exact in an f32"
    )]
    let rounded = if exact {
        (f64::from(v) * scale as f64).round() as i64
    } else {
        (v * scale as f32).round() as i64
    };
    rounded
}

/// `v`, clamped to [`lo`, `hi`] and truncated toward zero; a quiet NaN is `nan`, a signalling
/// one `hi` ([`quiet`]). Metal converts the colour's own `f64` (measured: a 32-bit component
/// keeps digits an `f32` drops).
fn integer(v: f64, lo: f64, hi: f64, nan: i64) -> i64 {
    if v.is_nan() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the largest code, an integer in range of an i64"
        )]
        return if quiet(v) { nan } else { hi as i64 };
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "in range of an i64 by the clamp"
    )]
    let truncated = v.clamp(lo, hi).trunc() as i64;
    truncated
}

/// Whether the NaN `v` is quiet: its `f64`'s top mantissa bit set.
fn quiet(v: f64) -> bool {
    v.to_bits() & (1 << 51) != 0
}

/// The NaN `v`, the colour's `f64`, in a float of `bits` bits whose exponent is `exponent` bits,
/// as Metal's fill stores it: `v`'s sign, the exponent's ones, the quiet bit, and the top bits of
/// `v`'s mantissa, as many as the float's mantissa holds.
fn nan_code(v: f64, bits: u32, exponent: u32) -> u64 {
    let d = v.to_bits();
    let mantissa = bits - 1 - exponent;
    let sign = (d >> 63) << (bits - 1);
    sign | (ones(exponent) << mantissa)
        | (1 << (mantissa - 1))
        | ((d >> (52 - mantissa)) & ones(mantissa))
}

/// `x / 2^shift`, rounded to nearest even.
fn round_shift(x: u32, shift: u32) -> u32 {
    if shift >= 32 {
        return 0;
    }
    if shift == 0 {
        return x;
    }
    let (q, r, half) = (x >> shift, x & ((1 << shift) - 1), 1 << (shift - 1));
    if r > half || (r == half && q & 1 == 1) {
        q + 1
    } else {
        q
    }
}

/// `v` as a half float's bits: rounded to nearest even, past the largest half an infinity, a
/// NaN the quiet NaN of its sign ([`encode`] gives a colour's NaN its payload, [`nan_code`]).
fn half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    if exponent == 0 {
        // A float denormal is far below a half's least.
        return sign;
    }
    let full = mantissa | 0x80_0000;
    let magnitude = if exponent <= 112 {
        // A half denormal, in units of 2^-24: `full * 2^(exponent - 126)`.
        round_shift(full, 126 - exponent)
    } else {
        let rounded = round_shift(full, 13);
        let (rounded, exponent) = if rounded == 0x800 {
            (0x400, exponent + 1)
        } else {
            (rounded, exponent)
        };
        let biased = exponent - 112;
        if biased >= 31 {
            0x7c00
        } else {
            (biased << 10) | (rounded & 0x3ff)
        }
    };
    sign | u16::try_from(magnitude).unwrap_or(0x7c00)
}

/// The 8-bit sRGB code Metal stores for `v`: the code whose decoded value, in Metal's table of
/// them, is nearest `v`, a value halfway between two taking the greater, so a NaN, zero or a
/// negative value is 0 and one from 1 on 255. The table holds each code's value as an `f32`: for
/// codes 0 to 10, on the curve's linear segment (`code / 255` up to 0.04045), `code` times
/// `1 / 3294.6`, both `f32`s, in `f32` arithmetic; for the rest the `f32` nearest the curve's
/// `((code / 255 + 0.055) / 1.055)^2.4`.
fn srgb(v: f32) -> u64 {
    let v = f64::from(v);
    srgb_edges().iter().take_while(|&&m| v >= m).count() as u64
}

/// The least value of each of the sRGB codes 1 to 255 ([`srgb`]).
fn srgb_edges() -> &'static [f64; 255] {
    static EDGES: std::sync::OnceLock<[f64; 255]> = std::sync::OnceLock::new();
    EDGES.get_or_init(|| {
        let decoded = |code: u16| -> f64 {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the step is an f32 operation of Metal's, from the f64 constant"
            )]
            let value = if code <= 10 {
                f32::from(code) * (1.0 / 3294.6f64) as f32
            } else {
                ((f64::from(code) / 255.0 + 0.055) / 1.055).powf(2.4) as f32
            };
            f64::from(value)
        };
        // The least value of each of codes 1 to 255: halfway from the code below to it, exact
        // as the sum of two `f32`s.
        std::array::from_fn(|k| {
            let k = u16::try_from(k).unwrap_or(u16::MAX);
            f64::midpoint(decoded(k), decoded(k + 1))
        })
    })
}

/// The 10-bit extended-range code Metal stores for `v`, the code of value `(code - 384) / 510`:
/// `v` plus `384 / 510`, times 510, each an `f32` operation, plus a half, rounded down and
/// clamped to [0, 1023]; a NaN is 0.
fn extended(v: f32) -> u64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the offset is the f32 nearest 384 / 510"
    )]
    let low = (384.0 / 510.0f64) as f32;
    let scaled = (v + low) * 510.0;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "in range of a u64 by the clamp; a NaN casts to 0"
    )]
    let code = (f64::from(scaled) + 0.5).floor().clamp(0.0, 1023.0) as u64;
    code
}

/// `v` as an unsigned float of a 5-bit exponent (bias 15) and `mantissa` bits of mantissa, as
/// Metal packs RG11B10Float's: rounded to nearest even, but below 2^-14, the least normal value,
/// aligned to that exponent first with the bits shifted out dropped; past the largest finite
/// value the largest; an infinity the infinity, a NaN all ones, and zero or a negative value,
/// negative infinity included, 0.
fn small_float(v: f32, mantissa: u32) -> u64 {
    let shift = 23 - mantissa;
    let infinity = 0x1fu32 << mantissa;
    if v.is_nan() {
        return u64::from(infinity | ((1 << mantissa) - 1));
    }
    if v.is_sign_negative() || v == 0.0 {
        return 0;
    }
    if v.is_infinite() {
        return u64::from(infinity);
    }
    let bits = v.to_bits();
    let exponent = bits >> 23;
    // `v`'s bits with the exponent rebiased to 15, or below 2^-14 its mantissa, implicit bit and
    // all, shifted to that exponent's places.
    let aligned = if exponent < 113 {
        ((bits & 0x7f_ffff) | 0x80_0000)
            .checked_shr(113 - exponent)
            .unwrap_or(0)
    } else {
        bits - (112 << 23)
    };
    let code = (aligned + (1 << (shift - 1)) - 1 + ((aligned >> shift) & 1)) >> shift;
    u64::from(code.min(infinity - 1))
}

/// `color`'s red, green and blue as RGB9E5Float, as Metal packs them: each the colour's `f32`
/// clamped to [0, 65408], a NaN 0; the exponent shared by the three (bits 27 to 31), the largest
/// one's, at least 2^-16, once its top 9 bits are rounded to nearest, halves up; and each
/// component's 9-bit mantissa (bits 0, 9 and 18) the component over that exponent's step,
/// rounded to nearest, halves up.
fn shared_exponent(color: ClearColor) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Metal converts the colour's f64 to an f32 first"
    )]
    let clamped = |v: f64| {
        let v = v as f32;
        if v.is_nan() {
            0.0
        } else {
            v.clamp(0.0, 65408.0)
        }
    };
    let rgb = [
        clamped(color.red),
        clamped(color.green),
        clamped(color.blue),
    ];
    let largest = rgb[0].max(rgb[1]).max(rgb[2]).max(2f32.powi(-16));
    // The largest's biased exponent once rounded to 9 bits: 111 to 142.
    let exponent = (largest.to_bits() + 0x4000) >> 23;
    // 2^(24 - shared exponent), whose product with a component is exact.
    let scale = f32::from_bits(0x8300_0000 - (exponent << 23));
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "at most 511 by the exponent's rounding"
    )]
    let mantissa = |v: f32| (f64::from(v * scale) + 0.5).floor() as u32;
    mantissa(rgb[0]) | (mantissa(rgb[1]) << 9) | (mantissa(rgb[2]) << 18) | ((exponent - 111) << 27)
}

/// The component `v` (the colour's `f64`, and `f32` its `f32`) encoded as `code`, in the code's
/// low bits.
fn encode(v: f64, f32: f32, code: Code) -> u64 {
    match code {
        Code::Unorm(bits) => {
            normalized(f32, 0.0, 1.0, ones(bits), matches!(bits, 4 | 8)).cast_unsigned()
        }
        Code::Snorm(bits) => {
            normalized(f32, -1.0, 1.0, ones(bits - 1), false).cast_unsigned() & ones(bits)
        }
        Code::Uint(32) if v > f64::from(i32::MAX) && v < 2_147_483_648.0 => 1 << 31,
        #[expect(
            clippy::cast_precision_loss,
            reason = "the largest code of at most 32 bits, exact in an f64"
        )]
        Code::Uint(bits) => integer(v, 0.0, ones(bits) as f64, 0).cast_unsigned(),
        Code::Sint(bits) => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the largest code of at most 31 bits, exact in an f64"
            )]
            let max = ones(bits - 1) as f64;
            let least = -(max + 1.0);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the least code, an integer in range of an i64"
            )]
            let nan = least as i64;
            integer(v, least, max, nan).cast_unsigned() & ones(bits)
        }
        Code::Half if v.is_nan() => nan_code(v, 16, 5),
        Code::Half => u64::from(half(f32)),
        Code::Float if v.is_nan() => nan_code(v, 32, 8),
        Code::Float => u64::from(f32.to_bits()),
        Code::Srgb => srgb(f32),
        Code::Extended => extended(f32),
        Code::ExtendedAlpha => extended(if f32.is_nan() {
            0.0
        } else {
            f32.clamp(0.0, 1.0)
        }),
        Code::SmallFloat(mantissa) => small_float(f32, mantissa),
    }
}

/// The formats Metal packs no colour in, for which a colour fill writes bytes Metal holds from
/// earlier work (the module's docs).
pub const LEFTOVER_FORMATS: [PixelFormat; 10] = [
    PixelFormat::Invalid,
    PixelFormat::Depth16Unorm,
    PixelFormat::Depth32Float,
    PixelFormat::Stencil8,
    PixelFormat::Depth24UnormStencil8,
    PixelFormat::Depth32FloatStencil8,
    PixelFormat::X32Stencil8,
    PixelFormat::X24Stencil8,
    PixelFormat::RGB10A82PXR10,
    PixelFormat::RGB10A82PXR10SRGB,
];

/// The pixel formats whose colour the runtime's fill packs, not this module: `R16Bfloat`,
/// `RG16Bfloat`, `RGBA16Bfloat`, `BGRA10_XR_sRGB`, `BGR10_XR_sRGB` and the RGBX formats ([`RGBX`];
/// the module's docs).
pub const RUNTIME_PACKED: [PixelFormat; 16] = [
    PixelFormat::R16Bfloat,
    PixelFormat::RG16Bfloat,
    PixelFormat::RGBA16Bfloat,
    PixelFormat::BGRA10XRSRGB,
    PixelFormat::BGR10XRSRGB,
    RGBX[0],
    RGBX[1],
    RGBX[2],
    RGBX[3],
    RGBX[4],
    RGBX[5],
    RGBX[6],
    RGBX[7],
    RGBX[8],
    RGBX[9],
    RGBX[10],
];

/// The RGBX formats, whose colour the runtime's fill packs ([`RUNTIME_PACKED`]).
pub const RGBX: [PixelFormat; 11] = [
    PixelFormat::RGBX8Unorm,
    PixelFormat::RGBX8UnormSRGB,
    PixelFormat::RGBX8Snorm,
    PixelFormat::RGBX8Uint,
    PixelFormat::RGBX8Sint,
    PixelFormat::RGBX16Uint,
    PixelFormat::RGBX16Sint,
    PixelFormat::RGBX16Float,
    PixelFormat::RGBX32Uint,
    PixelFormat::RGBX32Sint,
    PixelFormat::RGBX32Float,
];

/// The format of [`RUNTIME_PACKED`] the pixel format `raw` is, if any.
pub fn runtime_packed(raw: u64) -> Option<PixelFormat> {
    let format = PixelFormat::try_from(u32::try_from(raw).ok()?).ok()?;
    RUNTIME_PACKED.contains(&format).then_some(format)
}

/// `color` packed in the pixel format `raw`, as `fillTexture:...color:pixelFormat:` writes it,
/// for `sel`. A format this platform packs no colour in dies: one of [`LEFTOVER_FORMATS`], for
/// which Metal writes bytes of its own state, and any other (the formats nothing measured; a
/// value no format is). The formats of [`RUNTIME_PACKED`] die here too: the blit encoder sends
/// them to the runtime's colour fill before.
pub fn color_bytes(sel: &'static str, color: ClearColor, raw: u64) -> Vec<u8> {
    let format = u32::try_from(raw)
        .ok()
        .and_then(|raw| PixelFormat::try_from(raw).ok());
    let Some(packed) = format.and_then(|format| pack(format, color)) else {
        let why = if format.is_some_and(|f| LEFTOVER_FORMATS.contains(&f)) {
            "for which Metal writes bytes left by earlier work, not the colour's"
        } else {
            "which this platform does not pack as Metal does"
        };
        unrepresentable(sel, &format!("a colour in pixel format {raw}, {why}"));
    };
    packed
}

/// `color` packed in `format` as Metal packs it; `None` for a format this platform packs no
/// colour in.
fn pack(format: PixelFormat, color: ClearColor) -> Option<Vec<u8>> {
    if format == PixelFormat::RGB9E5Float {
        return Some(shared_exponent(color).to_le_bytes().to_vec());
    }
    let (fields, bytes) = layout(format)?;
    let components = [color.red, color.green, color.blue, color.alpha];
    let packed = fields.iter().fold(0u128, |packed, &(component, code, at)| {
        let v = components[component];
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Metal converts the colour's f64 to an f32 for the float formats"
        )]
        let encoded = encode(v, v as f32, code);
        packed | (u128::from(encoded) << at)
    });
    Some(packed.to_le_bytes()[..bytes].to_vec())
}

/// What a fill can write in a texture: its type, its first level's extent, its levels and its
/// array length.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// `textureType`.
    pub texture_type: TextureType,
    /// `width`, `height` and `depth`.
    pub extent: [u64; 3],
    /// `mipmapLevelCount`.
    pub levels: u64,
    /// `arrayLength`.
    pub array_length: u64,
}

/// Where a fill writes: the slice, the level, the origin and the size, inside the level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Place {
    /// The slice (a cube's face).
    pub slice: u64,
    /// The level.
    pub level: u64,
    /// The first texel.
    pub origin: [u64; 3],
    /// The texels across, down and deep.
    pub size: [u64; 3],
}

/// Where a fill of `region` at `level` and `slice` writes in a texture of `shape`, as Metal
/// writes it; `None` where it writes nothing. A multisample texture or a texture buffer, and a
/// region past 2^32 texels, die for `sel`, as nothing measured them.
pub fn place(
    sel: &'static str,
    shape: &Shape,
    level: u64,
    slice: u64,
    region: MtlRegion,
) -> Option<Place> {
    use TextureType as T;
    let deep = shape.texture_type == T::Type3D;
    let slice = match shape.texture_type {
        T::Type1D | T::Type2D | T::Type3D => 0,
        T::Type1DArray | T::Type2DArray => (slice < shape.array_length).then_some(slice)?,
        T::Cube => slice % 6,
        T::CubeArray => (slice < shape.array_length.saturating_mul(6)).then_some(slice)?,
        other => unrepresentable(
            sel,
            &format!("a fill of a {other:?} texture, which nothing measured"),
        ),
    };
    let (z, depth) = if deep {
        (region.origin.z, region.size.depth)
    } else {
        (0, 1)
    };
    let texels = region
        .size
        .width
        .checked_mul(region.size.height)
        .and_then(|area| area.checked_mul(depth));
    if texels.is_none_or(|texels| texels > MEASURED_TEXELS) {
        unrepresentable(
            sel,
            &format!("a fill of the region {region:?}, past what was measured to end on Metal"),
        );
    }
    if level >= shape.levels {
        return None;
    }
    let extent = shape.extent.map(|e| (e >> level.min(63)).max(1));
    let extent = [extent[0], extent[1], if deep { extent[2] } else { 1 }];
    let origin = [region.origin.x, region.origin.y, z];
    let size = [region.size.width, region.size.height, depth];
    let mut cut = [0; 3];
    for i in 0..3 {
        if size[i] == 0 || origin[i] >= extent[i] {
            return None;
        }
        cut[i] = size[i].min(extent[i] - origin[i]);
    }
    Some(Place {
        slice,
        level,
        origin,
        size: cut,
    })
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::types::{MtlOrigin, MtlSize};
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;

    /// The texels repeat their pattern from its start; a colour in a format Metal packs no
    /// colour in, or one not reproduced or measured (BGRA10_XR_sRGB), dies
    #[test]
    fn patterns_and_texels() {
        assert_eq!(texel(&[1, 2, 3, 4, 5], 4), [1, 2, 3, 4]);
        assert_eq!(texel(&[0x80], 4), [0x80; 4]);
        assert_eq!(texel(&[1, 2], 8), [1, 2, 1, 2, 1, 2, 1, 2]);
        let red = ClearColor {
            red: 1.0,
            green: 0.5,
            blue: 0.25,
            alpha: 1.0,
        };
        assert_eq!(color_bytes("s", red, 70), [0xff, 0x80, 0x40, 0xff]);
        assert_eq!(color_bytes("s", red, 71), [0xff, 0xbc, 0x89, 0xff]);
        assert_eq!(color_bytes("s", red, 92), 0x681c_03c0u32.to_le_bytes());
        assert_eq!(color_bytes("s", red, 93), 0x8101_0100u32.to_le_bytes());
        for format in [0, 553, 555, 252, 260, 9999] {
            let site = catch_death(|| color_bytes("s", red, format)).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, "s"));
        }
    }

    /// No format any Metal level the runtime measured describes holds a block of more bytes
    /// than a fill's copy of its pattern keeps ([`MOST_PATTERN_BYTES`]), so every texel a fill
    /// writes is cut from bytes it copied
    #[test]
    fn no_block_is_longer_than_a_fills_pattern() {
        let most = u64::try_from(MOST_PATTERN_BYTES).unwrap();
        for level in metal2vulkan_runtime::MetalLevel::ALL {
            for raw in 0..2048 {
                let [bytes, ..] = level.pixel_format_info(raw).words;
                assert!(bytes <= most, "{level:?} format {raw}: {bytes} bytes");
            }
        }
    }

    /// Where a fill writes, as measured on Metal: levels past the texture's write nothing; the
    /// region is cut at the level; 1D and 2D ignore the slice and every type but 3D the region's
    /// depth; arrays write their slices; a cube writes face `slice % 6`
    #[test]
    fn places_are_metals() {
        let shape = |texture_type, extent, levels, array_length| Shape {
            texture_type,
            extent,
            levels,
            array_length,
        };
        let r = |o: [u64; 3], s: [u64; 3]| {
            MtlRegion::new(
                MtlOrigin::new(o[0], o[1], o[2]),
                MtlSize::new(s[0], s[1], s[2]),
            )
        };
        let place_of =
            |shape: &Shape, level, slice, region| place("s", shape, level, slice, region);
        let at = |slice, level, origin, size| {
            Some(Place {
                slice,
                level,
                origin,
                size,
            })
        };
        let flat = shape(TextureType::Type2D, [4, 4, 1], 2, 1);
        assert_eq!(
            place_of(&flat, 0, 5, r([3, 0, 5], [4, 1, 0])),
            at(0, 0, [3, 0, 0], [1, 1, 1])
        );
        assert_eq!(
            place_of(&flat, 1, 0, r([1, 0, 0], [4, 4, 1])),
            at(0, 1, [1, 0, 0], [1, 2, 1])
        );
        assert_eq!(place_of(&flat, 2, 0, r([0, 0, 0], [1, 1, 1])), None);
        assert_eq!(place_of(&flat, 0, 0, r([4, 0, 0], [1, 1, 1])), None);
        assert_eq!(place_of(&flat, 0, 0, r([0, 0, 0], [0, 1, 1])), None);
        let array = shape(TextureType::Type2DArray, [2, 1, 1], 1, 2);
        assert_eq!(
            place_of(&array, 0, 1, r([0, 0, 1], [1, 1, 3])),
            at(1, 0, [0, 0, 0], [1, 1, 1])
        );
        assert_eq!(place_of(&array, 0, 2, r([0, 0, 0], [1, 1, 1])), None);
        let cube = shape(TextureType::Cube, [1, 1, 1], 1, 1);
        assert_eq!(
            place_of(&cube, 0, 100, r([0, 0, 0], [1, 1, 1])),
            at(4, 0, [0, 0, 0], [1, 1, 1])
        );
        let cubes = shape(TextureType::CubeArray, [1, 1, 1], 1, 2);
        assert_eq!(
            place_of(&cubes, 0, 7, r([0, 0, 0], [1, 1, 1])),
            at(7, 0, [0, 0, 0], [1, 1, 1])
        );
        assert_eq!(place_of(&cubes, 0, 12, r([0, 0, 0], [1, 1, 1])), None);
        let deep = shape(TextureType::Type3D, [2, 2, 2], 1, 1);
        assert_eq!(
            place_of(&deep, 0, 1, r([0, 0, 0], [1, 1, 3])),
            at(0, 0, [0, 0, 0], [1, 1, 2])
        );
        assert_eq!(place_of(&deep, 0, 0, r([0, 0, 0], [1, 1, 0])), None);
        let huge = r([0, 0, 0], [1 << 16, (1 << 16) + 1, 1]);
        let site = catch_death(|| place_of(&flat, 0, 0, huge)).unwrap_err();
        assert_eq!(site.file, "s");
        let many = shape(TextureType::Type2DMultisample, [4, 4, 1], 1, 1);
        assert!(catch_death(|| place_of(&many, 0, 0, r([0, 0, 0], [1, 1, 1]))).is_err());
        assert_eq!(
            place_of(&flat, 0, 0, r([0, 0, 0], [1 << 16, 1 << 16, 1])),
            at(0, 0, [0, 0, 0], [4, 4, 1])
        );
    }

    /// A colour or depth texture's texel is its description's block; a compressed (4:2:2
    /// included), stencil, combined depth/stencil or two-plane XR10 texture's fill is the
    /// runtime's
    #[test]
    fn texels_of_formats() {
        const FILL: &str = "fillTexture:level:slice:region:color:pixelFormat:";
        assert_eq!(texel_bytes(FILL, PixelFormat::RGBA8Unorm), Some(4));
        assert_eq!(texel_bytes(FILL, PixelFormat::Depth32Float), Some(4));
        assert_eq!(texel_bytes(FILL, PixelFormat::RGBA32Float), Some(16));
        for format in [
            PixelFormat::BC1RGBA,
            PixelFormat::GBGR422,
            PixelFormat::Stencil8,
            PixelFormat::X32Stencil8,
            PixelFormat::Depth32FloatStencil8,
            PixelFormat::RGB10A82PXR10,
            PixelFormat::RGB10A82PXR10SRGB,
        ] {
            assert_eq!(texel_bytes(FILL, format), None, "{format:?}");
        }
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
