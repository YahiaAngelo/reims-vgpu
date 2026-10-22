//! Apple's serialized pipeline descriptors (testing.md section 6.18): the reader the three
//! `new…PipelineDescriptorWithSerializedData:deserializationContext:` methods share, and the
//! decoders of each form into the translator's descriptor. On the Metal platform, Metal's private
//! initializers decode them; here nothing would, so this module does, as they do.
//!
//! **The format.** testing.md 6.18 gives it, and the rest was measured on this host (macOS 26.7,
//! 25G229) by deserializing blobs with Metal and serializing the result again:
//! - A block is a u8 count, then that many entries `{u8 key, u8 size, size bytes of value}`.
//!   Metal's serializer writes every value in 4 bytes, little-endian, and writes a key only for a
//!   value other than the property's default.
//! - **Entries are read in order**, each when it is reached: a function is resolved when its
//!   entry is read, so a blob cut short after a function's entry resolves it, then aborts. A key
//!   read twice is set twice: its last value wins, and a function's is resolved again.
//! - A nested block, an array or a string lies at its value's offset from its own block's start.
//!   An array is a u32 count, then each element's u32 offset from the array's start; each element
//!   is a block. An offset of 0 is nil: a nested block or array at 0 is none, an element at 0
//!   holds nothing, and a string at 0 aborts.
//! - A string is NUL-terminated; one with no NUL before the blob's end aborts.
//! - **An element's entries** set the element its key 0 selected last, by index; an entry before
//!   any index sets nothing.
//! - A boolean is any value but 0 (2, 3, 0x100 and 0x101 are each true, for every boolean key).
//! - Bytes past the blob's structures are ignored: core hands the compute and render forms the
//!   record's remaining bytes, zeros past descriptorLength.
//!
//! **What Metal aborts on** ends the process at the deserializer's selector ([`metal_aborts`]), in
//! every setup, as Metal's abort does: a block's count, an entry's header, a value, an array's
//! count or offsets, or a nested object past the blob's end (the suite's garbage and truncated
//! blobs, `compute_death_blob_*`, `render_objects_malformed_*`); a string at offset 0 or with no
//! NUL; an index past Metal's arrays (buffers and vertex or stage-input attributes and layouts
//! past 30, colour attachments past 7); and a compute function ref that resolves to nil, which
//! Metal aborts on as soon as the resolver answers. A render descriptor's vertex function or a tile
//! descriptor's function that is nil (a ref that resolves to nil, or no entry) is taken by Metal's
//! deserializer and aborts the state's creation; a compute descriptor with no function entry
//! does too. The translator's descriptors cannot be without that function, so these die when
//! the blob's last entry has been read. Nothing either form does between the two sends is
//! observable: core sets the resource index, then makes the state. A pixel format the runtime
//! holds raw is taken, as Metal's deserializer takes it, and dies at the state's creation
//! ([`RawFormat`]), but a tile descriptor's colour attachment of a value Metal's
//! `setPixelFormat:` does not take, which Metal aborts on as its entry is read.
//!
//! **Keys** are named by serializing each property of Metal's descriptor classes (their private
//! ones included) one at a time ([`compute`], [`render`] and [`tile`] list each form's):
//! - **A key no block defines is ignored**, as Metal's deserializer ignores it: for every
//!   undefined key of every block, values 1 and `0xffffffff` changed nothing the descriptor
//!   answers or serializes again. The entry is kept as the blob had it ([`Ignored`]), the
//!   first [`KEPT_IGNORED`] of a blob alone.
//! - **A key whose property the runtime has no field for** ([`Spi`]: Metal's SPI properties,
//!   compute 12, render 24, 27, 28 and 45 to 58) is kept, the last value read winning, and
//!   decided once its block has been read, as a missing function is: it dies naming its
//!   property ([`unrepresentable`]) where Metal's state would draw or write otherwise than one
//!   the runtime makes without it, and decodes to nothing elsewhere ([`SpiRule`],
//!   [`render_spi`]). A later entry's function is resolved first, so the resolver's raise stays
//!   the guest's, as on Metal. Measured by serializing each property alone, then by drawing
//!   under it, one case per process:
//!   - **Defaults**, for which the serializer writes no entry: 0, but `sampleCoverage` (27),
//!     1.0; `sampleMask` (28), all 64 bits; `clipDistanceEnableMask` (48), `0xff`; and
//!     `explicitVisibilityGroupID` (58), `0xffffffff`. A value is written in 4 bytes, a float
//!     as its bits. `sampleCoverageInvert` has no key: the serializer writes none for it.
//!   - **Inert at every value**: `clipDistanceEnableMask` (masks 0, 1, 2 and `0xff` clip alike,
//!     under vertex functions that write no clip distance, one and two),
//!     `depthStencilWriteDisabled` (at 1, depth and stencil writes land, a fragment function's
//!     depth among them, a discarded fragment writes neither, and the visibility count is
//!     unchanged).
//!   - **`sampleMask`** drops the samples it has no bit for: 0 and 2 draw nothing at 1 sample,
//!     5 half the samples at 4. One with a bit for each of the state's raster samples draws as
//!     the default does, and decodes to nothing.
//!   - **The alpha test** (45, 46): off, its function changes nothing; on, Always (7) draws
//!     every fragment at reference values 0, 0.5 and 1, and Never (0) draws none. On under
//!     another function dies.
//!   - **The logic operation** (47, 55): off, the operation changes nothing; on, 2 writes the
//!     fragment's colour as no operation does, under a partial write mask too, where no colour
//!     attachment blends, and writes it unblended where one does; 0 writes zero, `0xa` the
//!     colour's exclusive or with the attachment's. On dies but at 2 with nothing blending.
//!   - **The rest die at every value but the default**: `sampleCoverage` (0.5 draws nothing at
//!     1 sample, half the samples at 4), `pointSmoothEnabled` (a triangle strip draws
//!     nothing), `pointCoordLowerLeft` and `pointSizeOutputVS` (triangles unchanged, points
//!     unmeasured), `twoSideEnabled` (a fragment's `[[front_facing]]` and inputs unchanged, what
//!     it changes unmeasured), the depth comparisons' clamp masks, `vertexAmplificationMode`,
//!     `explicitVisibilityGroupID` and `needsCustomBorderColorSamplers` (render 57, compute 12),
//!     though at 1 a sampler of raw border colour 0, 1, 2, 3 or 7 samples past the edge as at 0,
//!     in a fragment function and in a kernel: no draw of this crate compares it yet.
//!
//!   The host's OpenGL on Metal (`4.1 Metal - 90.5`) sets `logicOperation` 2, `sampleMask`
//!   `0xffffffff` and, where its depth and stencil writes are off, `depthStencilWriteDisabled` 1
//!   on every pipeline it makes; each decodes to nothing.
//! - **`textureWriteRoundingMode`** (compute 10, render 30, tile 8), how a state's texture writes
//!   round: the device supports 0 to 2 (`supportsTextureWriteRoundingMode:` answers no for 3), and
//!   one past them dies ([`ROUNDING_MODES`]). Measured on the M3 Max, writing 64K values (from -2
//!   to 2, past 1 and -1 by float steps, and past the largest half) into 22 formats (unorm, snorm,
//!   sRGB, half and other floats, the packed and the XR formats among them) with no mode and each
//!   mode: mode 1 writes as the default everywhere; a kernel's writes and a tile function's (its
//!   textures and its colour attachments, half-float ones included) are the same at every mode, and
//!   so are the half-float texture writes of a vertex function (rasterization on or off), an object
//!   function and a mesh function; a render state's half-float writes at mode 2, into its colour
//!   attachments, blended or not, and into the textures its fragment function writes, round to
//!   nearest even where the default rounds them toward zero (`1 + 3 * 2^-12` is `3c00` at the
//!   default and `3c01` at mode 2, the tie `1 + 3 * 2^-11` is `3c01` and `3c02`, and from 65520 up
//!   `7bff` and infinity), and its other writes, depth among them, are the same, but for the
//!   textures its fragment function writes of `RG11B10Float`, which Metal's mode 2 rounds
//!   otherwise than its default (the runtime's probe, on the M3 Max as on the M4), and of
//!   `RGB9E5Float` and `RGBX16Float`, which nothing measured at this mode. The runtime's writes
//!   are Metal's default, but for a render state of its
//!   `TextureWriteRoundingMode::NearestEven`, which a render blob of mode 2 ([`NEAREST_EVEN`])
//!   decodes to: its half-float colour attachments that store the fragment function's output as
//!   written (no blend, or one adding the source at one to the destination at zero) round to
//!   nearest even, as Metal's do, whether the function outputs a `float` or a `half` (byte for byte
//!   over the sweeps, unblended and one-and-zero, `R16Float`, `RG16Float` and `RGBA16Float`), of 1,
//!   2 or 4 samples: Metal's mode 2 rounds each sample of a multisampled attachment as a write of
//!   one sample, and its resolve does not take the mode, as the runtime's (measured at 2 and 4
//!   samples, fully covered and a quarter pixel in, resolved by the drawing pass and by a later one
//!   that loads the samples and draws nothing; the runtime's device rounds the average at a tie as
//!   at mode 0, which it announces as `device-rounded-resolve`; Metal's device here makes none
//!   of 8 samples, so nothing measured them). The rest of its rounding is not built, so a mode 2
//!   state is decided once the runtime made it, as Metal's compile answers a refusal with the
//!   `NSError` first ([`rounded_writes`]): one with a fragment function (none where rasterization
//!   is off) that may write a half-float colour attachment ([`MODE_2_ATTACHMENT_FORMATS`], of a
//!   write mask with a channel, at an index the function has an output for:
//!   `fragment_color_outputs`) of another sample count, whose resolve nothing measured at this
//!   mode, or blended otherwise, or
//!   whose alpha alpha to coverage reads, or that may write a texture (`fragment_writes_textures`)
//!   and supports indirect command buffers, whose commands' draws tell nothing of the textures,
//!   dies there. A half-float attachment no output names is never written, so nothing of it rounds,
//!   and its state is made. So is one whose fragment function may write a texture otherwise. The
//!   runtime takes no mode for a texture write, for every format: it rounds a half-float one
//!   toward zero, and an `RG11B10Float` one by the device's conversion. So a draw under such a
//!   state dies at its selector where the function may write a texture of a half-float format
//!   (`R16Float`, `RG16Float`, `RGBA16Float`, `RGBX16Float`), of `RG11B10Float` or of
//!   `RGB9E5Float` ([`MODE_2_TEXTURE_FORMATS`]), and runs where it writes only textures of other
//!   formats, which Metal's mode 2 writes as its default does ([`rounded_texture_writes`]). Every
//!   other state of a mode the device supports is made as at the default.
//! - **The label** (compute 2, render and tile 0) is read and checked as Metal reads it, and the
//!   state made from the descriptor takes it (`PipelineOptions::label`): a diagnostic, which no
//!   answer to the guest reads and the runtime compiles nothing of.
//! - **A render descriptor's fragment function** that resolves to nil is none, as Metal's is, and
//!   rasterization off drops it: the translator's descriptor holds none then. Its ref is resolved
//!   either way, as Metal resolves it.
//! - **Raw values** become the runtime's types: one the type cannot hold dies at the selector
//!   ([`crate::gpu::metal_value`]), as the crate decides it everywhere.
//! - **Metal's arrays hold every index**, the ones the blob never names at their defaults
//!   (measured: a stage-input or vertex layout steps per vertex, rate 1, stride 0). The
//!   translator's maps hold what the blob selects, and, for each attribute's buffer the blob gives
//!   no layout, that default, which the translator would otherwise refuse as a missing layout. A
//!   colour attachment, depth or stencil format left `Invalid` is none, as the translator holds
//!   it, and tessellation at Metal's defaults is none.
//!
//! Every value but a function's is plain data; the resolver is core's, called on this thread
//! under the caller's hold, and its raise ends the decode and is the caller's, as it is carried
//! through Metal's.
//!
//! **A render blob's bytes** are a debug line under this module's target
//! (`REIMS_VGPU_LOG=warn,reims_vgpu_vulkan::gpu::serialized=debug`), written before the decode
//! reads them ([`Blob`]): a state the runtime refuses later then shows its descriptor's shape
//! outright, an abort's too. The line is a diagnostic: it changes nothing the guest or the host
//! sees, and with the target's debug lines off nothing formats it.

use std::collections::BTreeMap;
use std::fmt;

use metal2vulkan::metal::{
    AttributeFormat, BlendFactor, BlendOperation, ColorWriteMask, IndexType, Mutability,
    PixelFormat, PrimitiveTopologyClass, TextureWriteRoundingMode, VertexFormat,
    VertexStepFunction,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor,
    StageInputAttributeDescriptor, StageInputDescriptor, StageInputLayoutDescriptor, StageSource,
    TessellationDescriptor, TileRenderPipelineDescriptor, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::ToUsize;
use reims_vgpu_core::platform::device::FunctionResolver;
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::rt::Hold;

use crate::answer::{metal_aborts, plain, unrepresentable};
use crate::gpu::metal_value;
use crate::gpu::resource::VulkanFunction;
use crate::gpu::writes::Slots;
use crate::platform::VulkanPlatform;

/// The compute form's selector.
pub const COMPUTE_SEL: &str =
    "newComputePipelineDescriptorWithSerializedData:deserializationContext:";

/// The render form's selector.
pub const RENDER_SEL: &str =
    "newRenderPipelineDescriptorWithSerializedData:deserializationContext:";

/// The tile form's selector.
pub const TILE_SEL: &str =
    "newTileRenderPipelineDescriptorWithSerializedData:deserializationContext:";

/// A serialized descriptor's debug line ([the module docs](self)): the selector, the blob's length,
/// then its bytes in lowercase hex, two digits each, unseparated.
pub struct Blob<'a>(pub &'static str, pub &'a [u8]);

impl fmt::Display for Blob<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Blob(sel, data) = self;
        write!(f, "{sel} {} bytes: ", data.len())?;
        data.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

/// The highest index of Metal's buffer, vertex attribute and layout, and stage-input attribute and
/// layout arrays (31 entries each): one past it aborts (measured).
pub const LAST_BUFFER_INDEX: u32 = 30;

/// The highest index of Metal's colour attachment arrays (8 entries, render and tile): one past it
/// aborts (measured).
pub const LAST_COLOR_INDEX: u32 = 7;

/// An SPI property of Metal's descriptor that the runtime has no field for, by its key, and what
/// a value of it does to a state the runtime makes without it ([`SpiRule`], the module docs).
#[derive(Debug, Clone, Copy)]
pub struct Spi {
    /// Its key.
    pub key: u8,
    /// Metal's property.
    pub property: &'static str,
    /// Which of its values decode to nothing, and which die.
    pub rule: SpiRule,
    /// What Metal does at a value that dies, which no state the runtime makes does: the death's
    /// reason.
    pub what: &'static str,
}

/// Which values of an SPI key decode to nothing (measured on this host: the module docs); every
/// other value dies naming its property, once its block has been read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiRule {
    /// Metal's default, the value its serializer writes no entry for: every other value dies.
    DiesBut(u32),
    /// Every value: none changed what Metal draws or writes.
    Inert,
    /// Decided with the block's other keys ([`render_spi`]).
    Decided,
}

const fn spi(key: u8, property: &'static str, rule: SpiRule, what: &'static str) -> Spi {
    Spi {
        key,
        property,
        rule,
        what,
    }
}

/// `MTLCompareFunctionAlways`, the alpha test's function under which every fragment passes
/// (measured: at reference values 0, 0.5 and 1, a fragment of alpha 0.5 is drawn).
pub const ALPHA_ALWAYS: u32 = 7;

/// The logic operation that writes the fragment's colour as it is (measured: on, it writes what
/// Metal writes with the operation off, where no colour attachment blends; 0 writes zero, `0xa`
/// the colour's exclusive or with the attachment's).
pub const LOGIC_COPY: u32 = 2;

/// The compute form's SPI keys.
pub const COMPUTE_SPI: &[Spi] = &[spi(
    12,
    "needsCustomBorderColorSamplers",
    SpiRule::DiesBut(0),
    "custom border colours for the kernel's samplers, which no draw compares yet",
)];

/// The render form's SPI keys, named by serializing each property alone.
pub const RENDER_SPI: &[Spi] = &[
    spi(
        24,
        "vertexAmplificationMode",
        SpiRule::DiesBut(0),
        "a vertex amplification mode the runtime has no field for",
    ),
    spi(
        27,
        "sampleCoverage",
        SpiRule::DiesBut(SAMPLE_COVERAGE_ALL),
        "a coverage fraction Metal masks the samples by (0.5 draws nothing at 1 sample, and \
         half the samples at 4)",
    ),
    spi(
        28,
        "sampleMask",
        SpiRule::Decided,
        "a mask that drops the samples it has no bit for (0 or 2 draws nothing at 1 sample, 5 \
         half the samples at 4)",
    ),
    spi(
        45,
        "alphaTestEnabled",
        SpiRule::Decided,
        "an alpha test that discards fragments by their alpha (Never draws nothing)",
    ),
    spi(46, "alphaTestFunction", SpiRule::Decided, ""),
    spi(
        47,
        "logicOperationEnabled",
        SpiRule::Decided,
        "a logic operation in place of the colour written and blending",
    ),
    spi(48, "clipDistanceEnableMask", SpiRule::Inert, ""),
    spi(
        49,
        "pointSmoothEnabled",
        SpiRule::DiesBut(0),
        "point smoothing, under which a triangle strip draws nothing",
    ),
    spi(
        50,
        "pointCoordLowerLeft",
        SpiRule::DiesBut(0),
        "a point coordinate origin, unmeasured for points, which a state's draws may be",
    ),
    spi(
        51,
        "pointSizeOutputVS",
        SpiRule::DiesBut(0),
        "a point size source, unmeasured for points, which a state's draws may be",
    ),
    spi(
        52,
        "twoSideEnabled",
        SpiRule::DiesBut(0),
        "two-sided colours, by a mechanism unmeasured (a fragment's [[front_facing]] and its \
         inputs are unchanged under it)",
    ),
    spi(
        53,
        "vertexDepthCompareClampMask",
        SpiRule::DiesBut(0),
        "a clamp of the vertex function's depth comparisons, unmeasured",
    ),
    spi(
        54,
        "fragmentDepthCompareClampMask",
        SpiRule::DiesBut(0),
        "a clamp of the fragment function's depth comparisons, unmeasured",
    ),
    spi(55, "logicOperation", SpiRule::Decided, ""),
    spi(56, "depthStencilWriteDisabled", SpiRule::Inert, ""),
    spi(
        57,
        "needsCustomBorderColorSamplers",
        SpiRule::DiesBut(0),
        "custom border colours for the functions' samplers, which no draw compares yet",
    ),
    spi(
        58,
        "explicitVisibilityGroupID",
        SpiRule::DiesBut(u32::MAX),
        "a visibility group the runtime has no field for",
    ),
];

/// `sampleCoverage`'s default, 1.0, as its serializer writes the float's bits.
pub const SAMPLE_COVERAGE_ALL: u32 = 0x3f80_0000;

/// What a render blob's SPI keys (`values`, by key, each the last value read) ask of a state of
/// `samples` raster samples, colour attachments `colors`: where Metal's state would draw or
/// write otherwise than one the runtime makes without them, it dies naming the property, in key
/// order ([`RENDER_SPI`]):
/// - a key of [`SpiRule::DiesBut`] at a value but Metal's default;
/// - `sampleMask` (28) without a bit for each of the state's samples (one sample for 0 or 1);
/// - `alphaTestEnabled` (45) on, under an `alphaTestFunction` (46) but [`ALPHA_ALWAYS`]: off,
///   the function changes nothing;
/// - `logicOperationEnabled` (47) on, under a `logicOperation` (55) but [`LOGIC_COPY`], or
///   where a colour attachment blends: Metal writes the colour unblended then. Off, the
///   operation changes nothing.
pub fn render_spi(
    values: &BTreeMap<u8, u32>,
    samples: u32,
    colors: &[ColorAttachmentDescriptor],
    raw: &[Option<u64>],
) {
    let value = |key: u8| values.get(&key).copied();
    decide_spi(
        (RENDER_SEL, "render"),
        RENDER_SPI,
        values,
        |spi, v| match spi.key {
            28 => {
                let samples = samples.max(1);
                let all = 1_u64.checked_shl(samples).map_or(u64::MAX, |bit| bit - 1);
                (u64::from(v) & all != all).then(|| format!(" over {samples} samples"))
            }
            45 => {
                let function = value(46).unwrap_or(0);
                (v != 0 && function != ALPHA_ALWAYS)
                    .then(|| format!(" under alphaTestFunction {function:#x}"))
            }
            47 => {
                let operation = value(55).unwrap_or(0);
                // A colour attachment of a format held raw is an attachment, as Metal's.
                let blends = colors.iter().zip(raw).any(|(c, raw)| {
                    (c.pixel_format != PixelFormat::Invalid || raw.is_some()) && c.blending_enabled
                });
                let blending = if blends {
                    ", a colour attachment blending"
                } else {
                    ""
                };
                (v != 0 && (operation != LOGIC_COPY || blends))
                    .then(|| format!(" under logicOperation {operation:#x}{blending}"))
            }
            // `alphaTestFunction` and `logicOperation`, decided with 45 and 47.
            _ => None,
        },
    );
}

/// Each of `values` that `table` names, in key order, for the deserializer `sel` of the block
/// `block`: one of [`SpiRule::DiesBut`] at another value dies, and one of [`SpiRule::Decided`]
/// dies where `decided` answers what more the death names.
fn decide_spi(
    (sel, block): (&'static str, &str),
    table: &[Spi],
    values: &BTreeMap<u8, u32>,
    decided: impl Fn(&Spi, u32) -> Option<String>,
) {
    for spi in table {
        let Some(&v) = values.get(&spi.key) else {
            continue;
        };
        let detail = match spi.rule {
            SpiRule::DiesBut(default) => (v != default).then(String::new),
            SpiRule::Inert => None,
            SpiRule::Decided => decided(spi, v),
        };
        if let Some(detail) = detail {
            unrepresentable(
                sel,
                &format!(
                    "{} {v:#x} (key {} of the {block} block){detail}: {}, which no state the \
                     runtime makes has",
                    spi.property, spi.key, spi.what
                ),
            );
        }
    }
}

/// The `textureWriteRoundingMode` values the device supports: 0, the default, 1, which writes as
/// the default does, and [`NEAREST_EVEN`] (the module docs).
pub const ROUNDING_MODES: std::ops::RangeInclusive<u32> = 0..=2;

/// The `textureWriteRoundingMode` under which a render state's half-float writes round to
/// nearest even, where Metal's default rounds them toward zero (the module docs): the runtime's
/// `TextureWriteRoundingMode::NearestEven`.
pub const NEAREST_EVEN: u32 = 2;

/// The colour attachment formats a render state of [`NEAREST_EVEN`] rounds to nearest even, the
/// half-float formats `R16Float`, `RG16Float` and `RGBA16Float`: the runtime rounds them so where
/// the attachment stores the fragment function's output as written, and the rest dies as the
/// state is made ([`rounded_writes`]); the attachments of the other formats measured are written
/// alike at every mode (the module docs).
pub const MODE_2_ATTACHMENT_FORMATS: [PixelFormat; 3] = [
    PixelFormat::R16Float,
    PixelFormat::RG16Float,
    PixelFormat::RGBA16Float,
];

/// The formats of the textures a fragment function writes that a render state of
/// [`NEAREST_EVEN`] may write otherwise than the runtime, which takes no mode for a texture write
/// (the module docs): the half-float formats `R16Float`, `RG16Float`, `RGBA16Float` and
/// `RGBX16Float`, which the runtime rounds toward zero, where Metal's mode 2 rounds the first
/// three to nearest even (`RGBX16Float`'s are not measured); `RG11B10Float`, which the runtime
/// writes by the device's conversion, where Metal's mode 2 rounds it otherwise than its default;
/// and `RGB9E5Float`, which nothing measured at this mode. A draw that may write a texture of one
/// dies ([`rounded_texture_writes`]).
pub const MODE_2_TEXTURE_FORMATS: [PixelFormat; 6] = [
    PixelFormat::R16Float,
    PixelFormat::RG16Float,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBX16Float,
    PixelFormat::RG11B10Float,
    PixelFormat::RGB9E5Float,
];

/// The runtime's rounding of a render blob's `textureWriteRoundingMode` `mode`, one of
/// [`ROUNDING_MODES`]: nearest even at [`NEAREST_EVEN`], else toward zero, Metal's default.
fn texture_write_rounding(mode: u32) -> TextureWriteRoundingMode {
    match mode {
        NEAREST_EVEN => TextureWriteRoundingMode::NearestEven,
        _ => TextureWriteRoundingMode::TowardZero,
    }
}

/// Whether colour attachment `c` stores what the fragment function writes as written: it blends
/// nothing, or blends by adding the source at factor one to the destination at factor zero, in
/// its colour and its alpha alike.
fn stores_as_written(c: &ColorAttachmentDescriptor) -> bool {
    let replace = (BlendFactor::One, BlendFactor::Zero, BlendOperation::Add);
    !c.blending_enabled
        || ((
            c.source_rgb_blend_factor,
            c.destination_rgb_blend_factor,
            c.rgb_blend_operation,
        ) == replace
            && (
                c.source_alpha_blend_factor,
                c.destination_alpha_blend_factor,
                c.alpha_blend_operation,
            ) == replace)
}

/// What a render state of [`NEAREST_EVEN`], made of `desc`, writes that Metal rounds to nearest
/// even and the runtime does not, `None` where its writes are the runtime's (the module docs).
/// The runtime rounds a half-float colour attachment's channels to nearest even where the
/// attachment stores the fragment function's output as written ([`stores_as_written`]), as
/// Metal does, of a state of one sample or of a count of [`RESOLVE_MEASURED`], whose resolve Metal
/// takes from the samples so rounded, as the runtime does (the module docs); with a fragment
/// function that rasterizes (with rasterization off none runs), these stay:
/// - a half-float colour attachment whose write mask has a channel to write and whose index the
///   function has an output for (`color_outputs`, the state's `fragment_color_outputs`), of a
///   state of another sample count, whose resolve nothing measured at this mode
///   ([`resolve_measured`]);
/// - one that blends otherwise, whose result Metal rounds to nearest even and the runtime's
///   device converts by its own rounding;
/// - `RGBA16Float` at colour 0 under alpha to coverage, of a write mask with alpha, whose alpha
///   the runtime leaves to the device's conversion;
/// - a texture where the function may write one and the state supports indirect command buffers
///   (`commands_write_textures`: the descriptor's `supportIndirectCommandBuffers` and the
///   state's `fragment_writes_textures`), which the runtime does not round at this mode: an
///   indirect command's draw runs it with no draw here to tell the textures it writes. A state
///   without that support is made, and its draws decide its texture writes
///   ([`rounded_texture_writes`]). This is the narrowing that remains: Metal makes such a state,
///   and one that writes only textures of other formats runs as Metal's would; deciding it at an
///   execution needs what its commands' argument buffers hold.
///
/// An attachment no output names is never written, so nothing of it rounds.
pub fn rounded_writes(
    desc: &RenderPipelineDescriptor,
    color_outputs: &[u32],
    commands_write_textures: bool,
) -> Option<String> {
    if desc.fragment.is_none() || !desc.rasterization_enabled {
        return None;
    }
    let half = desc
        .color_attachments
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            u32::try_from(*index).is_ok_and(|index| color_outputs.contains(&index))
        })
        .find_map(|(index, c)| {
            let c = c.filter(|c| {
                MODE_2_ATTACHMENT_FORMATS.contains(&c.pixel_format)
                    && c.write_mask != ColorWriteMask::NONE
            })?;
            let coverage_alpha = index == 0
                && desc.alpha_to_coverage
                && c.pixel_format == PixelFormat::RGBA16Float
                && c.write_mask.contains(ColorWriteMask::ALPHA);
            let samples = desc.raster_sample_count;
            let why = if !resolve_measured(samples) {
                format!(
                    "of {samples} samples, whose resolve at this mode nothing measured (Metal's \
                     was measured at 1 sample and at each of {RESOLVE_MEASURED:?})"
                )
            } else if !stores_as_written(&c) {
                "blended with factors or operations other than one, zero and add, whose result \
                 the runtime's device converts by its own rounding"
                    .into()
            } else if coverage_alpha {
                "whose alpha alpha to coverage reads, which the runtime leaves to the device's \
                 conversion"
                    .into()
            } else {
                return None;
            };
            Some(format!(
                "colour attachment {index} of {}, {why}",
                c.pixel_format.name()
            ))
        });
    half.or_else(|| {
        commands_write_textures.then(|| {
            "a fragment function that may write a texture, of a state that supports indirect \
             command buffers, whose indirect commands' draws tell nothing of the textures it \
             writes, and whose half-float writes the runtime does not round at this mode"
                .into()
        })
    })
}

/// What a draw under a render state of [`NEAREST_EVEN`] whose fragment function may write a
/// texture (the state's `fragment_writes_textures`) may write that Metal's mode 2 writes
/// otherwise than the runtime, or that nothing measured at the mode, `None` where it writes no
/// texture of one of [`MODE_2_TEXTURE_FORMATS`] there (the module docs): `fragment`, the
/// function's slots by the state's reflection, or why it cannot tell them, and `formats`, the
/// fragment stage's texture slots holding a texture of one of [`MODE_2_TEXTURE_FORMATS`], by
/// Metal's name of its format (a view's own). These stay:
/// - a texture of one of those formats bound at a slot the function writes (`access::write` or
///   `access::read_write`);
/// - a function that reaches textures through handles (an argument buffer's), whose textures no
///   draw tells, or whose slots the reflection cannot tell;
/// - a function that writes no slot and takes no handles, whose texture writes go by a path the
///   reflection does not show.
///
/// A texture of any other format is written as Metal's default writes it, which Metal's mode 2
/// writes alike (measured: the module docs), so the draw runs.
pub fn rounded_texture_writes(
    fragment: Result<&Slots, &str>,
    formats: &BTreeMap<u64, &'static str>,
) -> Option<String> {
    let f = match fragment {
        Ok(f) => f,
        Err(why) => {
            return Some(format!(
                "a fragment function that may write textures, at slots its reflection cannot \
                 tell ({why})"
            ));
        }
    };
    if f.handles {
        return Some(
            "a fragment function that may write textures through handles (an argument \
             buffer's), whose formats no draw tells"
                .into(),
        );
    }
    if f.textures.is_empty() {
        return Some(
            "a fragment function that may write textures by a path its reflection does not \
             show (no texture slot it writes, and no handles)"
                .into(),
        );
    }
    formats
        .iter()
        .find(|&(&slot, _)| f.texture(slot))
        .map(|(slot, name)| {
            format!("texture {slot} of {name}, at a slot the fragment function writes")
        })
}

/// The sample counts above one at which a [`NEAREST_EVEN`] state's half-float colour attachment
/// that stores the fragment function's output as written was measured to resolve as Metal's (the
/// module docs): a state of another count above one dies as it is made ([`rounded_writes`]).
const RESOLVE_MEASURED: [u32; 2] = [2, 4];

/// Whether a state of `samples` samples resolves its half-float attachments as Metal's at
/// [`NEAREST_EVEN`]: one sample (0 is taken as one), or a count of [`RESOLVE_MEASURED`]; under the
/// tests, also a count [`RESOLVE_PROBED`] holds, for the probe that measures it.
fn resolve_measured(samples: u32) -> bool {
    #[cfg(test)]
    if RESOLVE_PROBED.load(std::sync::atomic::Ordering::Relaxed) == samples {
        return true;
    }
    samples <= 1 || RESOLVE_MEASURED.contains(&samples)
}

/// A sample count [`resolve_measured`] takes beside [`RESOLVE_MEASURED`], for the probe that
/// measures its resolve against Metal's (`encode_render`'s
/// `metal_resolves_mode_2_at_other_sample_counts_here`); 0, none, outside it.
#[cfg(test)]
pub(crate) static RESOLVE_PROBED: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// The relay at `sel`, or its death in strict mode ([`crate::answer::inexact`]), of a draw or
/// an execution of indirect commands that inherit the encoder's state, under a render state of
/// [`NEAREST_EVEN`] whose fragment function may write `what` ([`rounded_texture_writes`]): the
/// runtime rounds those writes by its default.
pub fn rounded_texture_writes_inexact(sel: &'static str, what: &str) {
    crate::answer::inexact(
        sel,
        crate::answer::Inexact::WriteRounding,
        &format!(
            "a draw or execution under a state of textureWriteRoundingMode {NEAREST_EVEN:#x}, \
             with {what}: Metal's mode 2 rounds a fragment function's texture writes of \
             R16Float, RG16Float, RGBA16Float and RG11B10Float otherwise than its default, and \
             nothing measured RGBX16Float's and RGB9E5Float's, where the runtime takes no mode \
             for a texture write"
        ),
    );
}

/// The relay at `sel`, or its death in strict mode ([`crate::answer::inexact`]), of a render
/// state of [`NEAREST_EVEN`] that writes `what` ([`rounded_writes`]): the state is made, and
/// rounds those writes by the runtime's default.
pub fn rounded_writes_inexact(sel: &'static str, what: &str) {
    crate::answer::inexact(
        sel,
        crate::answer::Inexact::WriteRounding,
        &format!(
            "textureWriteRoundingMode {NEAREST_EVEN:#x} of a descriptor with {what}: Metal rounds \
             the fragment function's half-float writes there to nearest even, and the runtime \
             rounds to nearest even only an attachment that stores them as written"
        ),
    );
}

/// `MTLBufferLayoutStrideDynamic` (`NSUIntegerMax`) as the serializer writes it, in 4 bytes
/// (measured for vertex and stage-input layouts).
pub const DYNAMIC_STRIDE: u32 = u32::MAX;

/// The most entries of keys no block defines one decode keeps ([`Ignored`], [`Decoded::ignored`]),
/// the first in the blob's order: Metal reads none of them, and nothing here reads one but a test
/// or a debug line, so the rest are dropped. The elements of an array may all lie at one block,
/// so the entries a blob's elements reach grow with the elements times the block's entries: a
/// 1 MiB blob of aliased elements, each reaching 254 undefined keys of 255 bytes, would keep about
/// 18 GiB of copies, which ends the process, where Metal's deserializer allocates nothing for them.
pub const KEPT_IGNORED: usize = 64;

/// An entry of a key its block does not define, which Metal's deserializer ignores (measured: see
/// the module docs), kept as the blob had it. A key is this format's field, as a byte offset is a
/// struct's, so it shows as `field_xx`, `xx` the key in lowercase hex (AGENTS.md, "Never guess a
/// name").
#[derive(Clone, PartialEq, Eq)]
pub struct Ignored {
    /// The block it is in (`render.colorAttachments`).
    pub block: &'static str,
    /// Its key.
    pub key: u8,
    /// Its value's bytes, as far as the blob holds them: Metal never reads them.
    pub value: Vec<u8>,
}

impl fmt::Debug for Ignored {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.field_{:02x} = {:02x?}",
            self.block, self.key, self.value
        )
    }
}

/// A decoded descriptor, with the entries Metal ignores.
#[derive(Debug)]
pub struct Decoded<D> {
    /// The translator's descriptor.
    pub desc: D,
    /// The first [`KEPT_IGNORED`] entries of keys no block defines, in the blob's order.
    pub ignored: Vec<Ignored>,
    /// Whether a render blob's `textureWriteRoundingMode` is [`NEAREST_EVEN`], which its state
    /// is decided for once made ([`rounded_writes`]); `false` for every other blob.
    pub nearest_even: bool,
    /// The blob's label, if it has one: the label of a state made from the descriptor.
    pub label: Option<String>,
    /// The first pixel format the blob holds that the runtime keeps raw ([`RawFormat`]), which
    /// the state's creation dies at.
    pub raw_format: Option<RawFormat>,
}

/// A pixel format a render or tile blob holds that Metal describes no format for, kept raw
/// ([`crate::gpu::held_format`]; the runtime has every format Metal describes): Metal's
/// deserializer takes it, and its state's creation makes the state or asserts. Measured on the
/// M3 Max, one case a process, with `MTL_DEBUG_LAYER` unset
/// (`raw_pipeline_formats_die_where_states_are_made`):
/// - Metal makes the state of a render descriptor whose colour attachment is 576, 577, 578,
///   579, 501, 26, 556 or 624, of one whose depth attachment is `AnyDepth` (651) or stencil
///   attachment `AnyStencil` (652), and of a tile descriptor whose colour attachment is 576 to
///   579, 26 or 556 (the runtime has each of them and makes those tile states, and the render
///   states of 576 to 579; it refuses the others at a render descriptor's attachment, which
///   dies named at the creator, [`RawFormat::refused`]);
/// - it asserts at the state's creation where a render descriptor's colour, depth or stencil
///   attachment is a value it describes no format for (2, 255, 262, 630 or 2000: "pixelFormat,
///   for color render target(0), is not a valid MTLPixelFormat"), or a format it cannot render
///   to there (`AnyDepth` as a colour; 576 or `BGRA8Unorm` as a depth: "depthAttachmentPixelFormat
///   MTLPixelFormatR10Unorm_X6 is not depth renderable"), and where a tile descriptor's colour
///   attachment is 255 or 630;
/// - a tile descriptor's colour attachment of a value Metal's `setPixelFormat:` does not take
///   ([`crate::gpu::format_info::setter_takes`]) asserts in the deserializer
///   ("validateMTLPixelFormat:30: failed assertion `pixelFormat is not a valid MTLPixelFormat.'",
///   measured at 2000), which dies there.
///
/// The creator dies at the first, as Metal's assertion ([`RawFormat::dies`]); the death of a
/// format Metal cannot render to there is its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawFormat {
    /// The attachment that holds it: `colour attachment 0`, `depth attachment` or
    /// `stencil attachment`.
    pub attachment: String,
    /// The raw value.
    pub raw: u64,
}

impl RawFormat {
    /// The first of `colors` (by index), then `depth` and `stencil`, held raw.
    fn first(colors: &[Option<u64>; 8], depth: Option<u64>, stencil: Option<u64>) -> Option<Self> {
        let colour = colors.iter().enumerate().find_map(|(i, raw)| {
            raw.map(|raw| RawFormat {
                attachment: format!("colour attachment {i}"),
                raw,
            })
        });
        let other = |attachment: &str, raw: Option<u64>| {
            raw.map(|raw| RawFormat {
                attachment: attachment.to_owned(),
                raw,
            })
        };
        colour
            .or_else(|| other("depth attachment", depth))
            .or_else(|| other("stencil attachment", stencil))
    }

    /// Dies at `sel`, the state's creator, as Metal's assertion, where it describes no format
    /// for the value ([`crate::gpu::desc::lacked_as`]).
    pub fn dies(&self, sel: &'static str) -> ! {
        let what = format!("a state whose {} is", self.attachment);
        crate::gpu::desc::lacked_as(sel, &what, self.raw, None)
    }

    /// The runtime's refusal `e` of a state at `sel` whose attachments' pixel formats are
    /// `colors`, `depth` and `stencil`: a refusal of a format (`Feature::Format`) dies at the
    /// attachment the refusal names, where Metal describes its format, as [`RawFormat::dies`]
    /// does, named by Metal's description. One that names no attachment, of a state that holds a
    /// format Metal describes, dies naming none, its line the descriptor's formats and blend
    /// states, `formats` ([`crate::gpu::states::formats`]): the format refused may be any
    /// attachment's, or a function's. Any other refusal is `e`.
    pub fn refused(
        sel: &'static str,
        colors: &[Option<PixelFormat>],
        (depth, stencil): (Option<PixelFormat>, Option<PixelFormat>),
        formats: impl FnOnce() -> String,
        e: rt::Error,
    ) -> rt::Error {
        use metal2vulkan::error::Attachment;
        if !crate::gpu::desc::is_format_refusal(&e) {
            return e;
        }
        let raw = |f: Option<PixelFormat>| f.map(|f| u64::from(f.as_raw()));
        let mut held: Vec<(String, Option<u64>)> = colors
            .iter()
            .enumerate()
            .map(|(i, f)| (format!("colour attachment {i}"), raw(*f)))
            .collect();
        held.push(("depth attachment".to_owned(), raw(depth)));
        held.push(("stencil attachment".to_owned(), raw(stencil)));
        let named = match &e {
            rt::Error::Unsupported(u) => u.location.as_ref().and_then(|l| l.attachment()),
            _ => None,
        };
        let at = match named {
            Some(Attachment::Color(i)) => usize::try_from(i).ok(),
            Some(Attachment::Depth) => Some(colors.len()),
            Some(Attachment::Stencil) => Some(colors.len() + 1),
            _ => None,
        };
        let described = |(_, f): &&(String, Option<u64>)| {
            f.is_some_and(|raw| crate::gpu::format_info::described(sel, raw))
        };
        let found = match at {
            Some(i) => held.get(i).filter(|a| described(a)),
            None if held.iter().any(|a| described(&a)) => unrepresentable(
                sel,
                &format!(
                    "a state a pixel format of which the runtime refuses, at an attachment it \
                     does not name or in a function: {}; {}",
                    crate::answer::refusal(&e),
                    formats()
                ),
            ),
            None => None,
        };
        if let Some((attachment, Some(raw))) = found {
            let what = format!("a state whose {attachment} is");
            crate::gpu::desc::lacked_as(sel, &what, *raw, Some(&e));
        }
        e
    }
}

/// Core's resolver, reborrowed for one entry.
type Resolver<'r> = &'r mut dyn FnMut(Hold<'_>, u64) -> RaiseResult<Option<VulkanFunction>>;

/// One entry of a block, its value not read yet.
#[derive(Debug, Clone, Copy)]
struct Entry {
    key: u8,
    size: u8,
    /// Where its value's bytes start.
    value_at: usize,
}

/// A blob, read for the deserializer `sel`, which every death names.
#[derive(Clone, Copy)]
struct Reader<'a> {
    sel: &'static str,
    bytes: &'a [u8],
}

/// A block's entries, each header read when it is reached.
struct Entries<'a> {
    reader: Reader<'a>,
    block: &'static str,
    next: usize,
    left: u8,
}

impl Iterator for Entries<'_> {
    type Item = Entry;

    fn next(&mut self) -> Option<Entry> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        let at = self.next;
        let value_at = self.reader.end_within(at, 2, || {
            format!(
                "an entry's header past the blob's end, in the {} block",
                self.block
            )
        });
        let (key, size) = (self.reader.bytes[at], self.reader.bytes[at + 1]);
        self.next = value_at + usize::from(size);
        Some(Entry {
            key,
            size,
            value_at,
        })
    }
}

impl<'a> Reader<'a> {
    fn new(sel: &'static str, bytes: &'a [u8]) -> Reader<'a> {
        Reader { sel, bytes }
    }

    /// The end of the `length` bytes at `at`, which the blob holds; past its end, Metal aborts.
    fn end_within(&self, at: usize, length: usize, what: impl FnOnce() -> String) -> usize {
        at.checked_add(length)
            .filter(|&end| end <= self.bytes.len())
            .unwrap_or_else(|| metal_aborts(self.sel, &what()))
    }

    /// The u32 at `at`, little-endian.
    fn u32_at(&self, at: usize, what: impl FnOnce() -> String) -> u32 {
        let end = self.end_within(at, 4, what);
        let mut word = [0; 4];
        word.copy_from_slice(&self.bytes[at..end]);
        u32::from_le_bytes(word)
    }

    /// The entries of the block at `at`, `block` naming it.
    fn entries(self, at: usize, block: &'static str) -> Entries<'a> {
        self.end_within(at, 1, || format!("the {block} block past the blob's end"));
        Entries {
            reader: self,
            block,
            next: at + 1,
            left: self.bytes[at],
        }
    }

    /// The value of `e`, a key `block` defines: one past the blob's end aborts. Metal's serializer
    /// writes 4 bytes; Metal reads other sizes in more than one way (some abort, some read the
    /// low bytes), so this platform takes none.
    fn value(&self, block: &'static str, e: Entry) -> u32 {
        self.end_within(e.value_at, usize::from(e.size), || {
            format!(
                "key {}'s value past the blob's end, in the {block} block",
                e.key
            )
        });
        if e.size != 4 {
            unrepresentable(
                self.sel,
                &format!(
                    "a {}-byte value of key {} of the {block} block",
                    e.size, e.key
                ),
            );
        }
        self.u32_at(e.value_at, || {
            format!(
                "key {}'s value past the blob's end, in the {block} block",
                e.key
            )
        })
    }

    /// The boolean `e` holds: any value but 0 is true.
    fn flag(&self, block: &'static str, e: Entry) -> bool {
        self.value(block, e) != 0
    }

    /// The raw value of `e` as the Metal enum `T`, `what` naming it.
    fn metal<T: TryFrom<u32>>(&self, block: &'static str, e: Entry, what: &str) -> T {
        metal_value(self.sel, what, u64::from(self.value(block, e)))
    }

    /// The pixel format `e` holds, `Invalid` included: the runtime's, or `Invalid` and the raw
    /// value the runtime keeps raw ([`RawFormat`]).
    fn pixel_format(&self, block: &'static str, e: Entry) -> (PixelFormat, Option<u64>) {
        match crate::gpu::held_format(self.sel, u64::from(self.value(block, e))) {
            Ok(pf) => (pf, None),
            Err(raw) => (PixelFormat::Invalid, Some(raw)),
        }
    }

    /// Where the object `e` names lies, in the block or array at `base`: `None` for an offset of
    /// 0 (nil).
    fn nested(&self, base: usize, block: &'static str, e: Entry) -> Option<usize> {
        let offset = self.value(block, e);
        if offset == 0 {
            return None;
        }
        let at = base.saturating_add(offset.to_usize());
        self.end_within(at, 1, || {
            format!(
                "key {}'s object past the blob's end, in the {block} block",
                e.key
            )
        });
        Some(at)
    }

    /// The elements of the array at `at`: where each starts, `None` for an offset of 0.
    ///
    /// The count is the guest's: its offsets must lie in the blob, as Metal aborts at one past
    /// its end, and that is checked before anything is sized from it, so a count past what the
    /// blob holds (0xffffffff would size 64 GiB of elements) dies there, as every overrun of the
    /// reader does, never in the allocator.
    fn array(&self, at: usize, block: &'static str) -> Vec<Option<usize>> {
        let count = self
            .u32_at(at, || format!("the {block} array past the blob's end"))
            .to_usize();
        self.end_within(at + 4, count.saturating_mul(4), || {
            format!("the {block} array's offsets past the blob's end")
        });
        (0..count)
            .map(|i| {
                let offset = self.u32_at(at + 4 + 4 * i, || {
                    format!("the {block} array's offsets past the blob's end")
                });
                (offset != 0).then(|| {
                    let element = at.saturating_add(offset.to_usize());
                    self.end_within(element, 1, || {
                        format!("an element of the {block} array past the blob's end")
                    });
                    element
                })
            })
            .collect()
    }

    /// Each entry of each element of the array `e` names in the block at `base`, with the index
    /// its element's key 0 selected last; key 0 itself is passed too, the element it selects
    /// then being Metal's at its defaults if nothing set it before. An index past `last` aborts,
    /// and an entry before any index sets nothing.
    fn elements(
        &self,
        base: usize,
        block: &'static str,
        e: Entry,
        last: u32,
        mut entry: impl FnMut(u32, Entry),
    ) {
        let Some(array) = self.nested(base, block, e) else {
            return;
        };
        for element in self.array(array, block).into_iter().flatten() {
            let mut selected = None;
            for item in self.entries(element, block) {
                if item.key == 0 {
                    let index = self.value(block, item);
                    if index > last {
                        metal_aborts(self.sel, &format!("index {index} of the {block} array"));
                    }
                    selected = Some(index);
                }
                if let Some(index) = selected {
                    entry(index, item);
                }
            }
        }
    }

    /// The label `e` names in the block at `base`, read as Metal reads it (a string at 0, or with
    /// no NUL before the blob's end, aborts): its bytes to the NUL, as UTF-8 (a byte that is not
    /// is replaced: the label is a diagnostic).
    fn label(&self, base: usize, block: &'static str, e: Entry) -> String {
        let offset = self.value(block, e);
        let at = base.saturating_add(offset.to_usize());
        let tail = self.bytes.get(at..).filter(|_| offset != 0);
        let Some(text) = tail.and_then(|tail| {
            tail.split(|&b| b == 0)
                .next()
                .filter(|t| t.len() < tail.len())
        }) else {
            metal_aborts(
                self.sel,
                &format!("the {block} block's label at offset {offset}"),
            );
        };
        String::from_utf8_lossy(text).into_owned()
    }

    /// Keeps in `ignored` the entry `e` of a key `block` does not define, which Metal ignores, as
    /// the blob holds it, while `ignored` holds fewer than [`KEPT_IGNORED`].
    fn ignore(&self, ignored: &mut Vec<Ignored>, block: &'static str, e: Entry) {
        if ignored.len() >= KEPT_IGNORED {
            return;
        }
        let end = e.value_at.saturating_add(usize::from(e.size));
        let value = self
            .bytes
            .get(e.value_at..end.min(self.bytes.len()))
            .unwrap_or_default()
            .to_vec();
        ignored.push(Ignored {
            block,
            key: e.key,
            value,
        });
    }

    /// Key `e` of `block`: the SPI property among `table` it is, its value kept in `spi` (the
    /// last read wins), decided once the block is read; `false` for a key not among them.
    fn spi(
        &self,
        block: &'static str,
        e: Entry,
        table: &[Spi],
        spi: &mut BTreeMap<u8, u32>,
    ) -> bool {
        if !table.iter().any(|s| s.key == e.key) {
            return false;
        }
        spi.insert(e.key, self.value(block, e));
        true
    }

    /// `textureWriteRoundingMode` at `e`: one of [`ROUNDING_MODES`], which its form decides (the
    /// module docs); one the device does not support, which nothing measured, dies.
    fn rounding_mode(&self, block: &'static str, e: Entry) -> u32 {
        let value = self.value(block, e);
        if !ROUNDING_MODES.contains(&value) {
            unrepresentable(
                self.sel,
                &format!(
                    "textureWriteRoundingMode {value:#x} (key {} of the {block} block), which \
                     the device does not support",
                    e.key
                ),
            );
        }
        value
    }

    /// Key `e` of `block`: an SPI property among `table` ([`Reader::spi`]), else kept as Metal
    /// ignores it.
    fn other(
        &self,
        block: &'static str,
        e: Entry,
        (table, spi): (&[Spi], &mut BTreeMap<u8, u32>),
        ignored: &mut Vec<Ignored>,
    ) {
        if !self.spi(block, e, table, spi) {
            self.ignore(ignored, block, e);
        }
    }

    /// The function `e`'s ref names, from core's resolver under `hold`: `None` for one it resolves
    /// to nil. Its raise is the caller's.
    fn function(
        &self,
        hold: Hold<'_>,
        block: &'static str,
        e: Entry,
        resolve: Resolver<'_>,
    ) -> RaiseResult<Option<StageSource>> {
        let function_ref = self.value(block, e);
        let function = resolve(hold, u64::from(function_ref))?;
        Ok(function.map(|f| StageSource::Function(plain(|| f.function_descriptor()))))
    }

    /// The pipeline buffers' mutabilities (`buffers`, `vertexBuffers`, `fragmentBuffers`,
    /// `tileBuffers`): each element `{0 index, 1 mutability}`.
    fn buffers(
        &self,
        base: usize,
        block: &'static str,
        e: Entry,
        buffers: &mut BTreeMap<u32, Mutability>,
        ignored: &mut Vec<Ignored>,
    ) {
        self.elements(
            base,
            block,
            e,
            LAST_BUFFER_INDEX,
            |index, item| match item.key {
                0 => {}
                1 => {
                    buffers.insert(index, self.metal(block, item, "mutability"));
                }
                _ => self.ignore(ignored, block, item),
            },
        );
    }

    /// The stage-input descriptor (compute key 3) at `at`.
    fn stage_input(&self, at: usize, ignored: &mut Vec<Ignored>) -> StageInputDescriptor {
        const BLOCK: &str = "compute.stageInputDescriptor";
        const ATTRIBUTES: &str = "compute.stageInputDescriptor.attributes";
        const LAYOUTS: &str = "compute.stageInputDescriptor.layouts";
        let mut s = StageInputDescriptor::new();
        for e in self.entries(at, BLOCK) {
            match e.key {
                0 => self.elements(at, BLOCK, e, LAST_BUFFER_INDEX, |index, item| {
                    let a = s
                        .attributes
                        .entry(index)
                        .or_insert_with(unset_stage_input_attribute);
                    match item.key {
                        0 => {}
                        1 => a.format = self.metal(ATTRIBUTES, item, "attribute format"),
                        2 => a.offset = self.value(ATTRIBUTES, item),
                        3 => a.buffer_index = self.value(ATTRIBUTES, item),
                        _ => self.ignore(ignored, ATTRIBUTES, item),
                    }
                }),
                1 => self.elements(at, BLOCK, e, LAST_BUFFER_INDEX, |index, item| {
                    let l = s
                        .layouts
                        .entry(index)
                        .or_insert_with(unset_stage_input_layout);
                    match item.key {
                        0 => {}
                        1 => l.step_function = self.metal(LAYOUTS, item, "step function"),
                        2 => l.step_rate = self.value(LAYOUTS, item),
                        3 => l.stride = stride(self.value(LAYOUTS, item)),
                        _ => self.ignore(ignored, LAYOUTS, item),
                    }
                }),
                2 => s.index_type = self.metal::<IndexType>(BLOCK, e, "index type"),
                3 => s.index_buffer_index = self.value(BLOCK, e),
                _ => self.ignore(ignored, BLOCK, e),
            }
        }
        for buffer in s.attributes.values().map(|a| a.buffer_index) {
            s.layouts
                .entry(buffer)
                .or_insert_with(unset_stage_input_layout);
        }
        s
    }

    /// The vertex descriptor (render key 3) at `at`.
    fn vertex_descriptor(&self, at: usize, ignored: &mut Vec<Ignored>) -> VertexDescriptor {
        const BLOCK: &str = "render.vertexDescriptor";
        const ATTRIBUTES: &str = "render.vertexDescriptor.attributes";
        const LAYOUTS: &str = "render.vertexDescriptor.layouts";
        let mut v = VertexDescriptor::new();
        for e in self.entries(at, BLOCK) {
            match e.key {
                0 => self.elements(at, BLOCK, e, LAST_BUFFER_INDEX, |index, item| {
                    let a = v
                        .attributes
                        .entry(index)
                        .or_insert_with(unset_vertex_attribute);
                    match item.key {
                        0 => {}
                        1 => a.format = self.metal(ATTRIBUTES, item, "vertex format"),
                        2 => a.offset = self.value(ATTRIBUTES, item),
                        3 => a.buffer_index = self.value(ATTRIBUTES, item),
                        _ => self.ignore(ignored, ATTRIBUTES, item),
                    }
                }),
                1 => self.elements(at, BLOCK, e, LAST_BUFFER_INDEX, |index, item| {
                    let l = v.layouts.entry(index).or_insert_with(unset_vertex_layout);
                    match item.key {
                        0 => {}
                        1 => {
                            l.step_function = self.metal::<VertexStepFunction>(
                                LAYOUTS,
                                item,
                                "vertex step function",
                            );
                        }
                        2 => l.step_rate = self.value(LAYOUTS, item),
                        3 => l.stride = stride(self.value(LAYOUTS, item)),
                        _ => self.ignore(ignored, LAYOUTS, item),
                    }
                }),
                _ => self.ignore(ignored, BLOCK, e),
            }
        }
        for buffer in v.attributes.values().map(|a| a.buffer_index) {
            v.layouts.entry(buffer).or_insert_with(unset_vertex_layout);
        }
        v
    }

    /// The colour attachments (render key 8) into `colors`: each element `{0 index, 1
    /// pixelFormat, 2 blendingEnabled, 3 sourceRGBBlendFactor, 4 destinationRGBBlendFactor, 5
    /// rgbBlendOperation, 6 sourceAlphaBlendFactor, 7 destinationAlphaBlendFactor, 8
    /// alphaBlendOperation, 9 writeMask}`.
    fn color_attachments(
        &self,
        e: Entry,
        (colors, raw): (&mut [ColorAttachmentDescriptor; 8], &mut [Option<u64>; 8]),
        ignored: &mut Vec<Ignored>,
    ) {
        const BLOCK: &str = "render.colorAttachments";
        self.elements(0, BLOCK, e, LAST_COLOR_INDEX, |index, item| {
            let c = &mut colors[index.to_usize()];
            match item.key {
                0 => {}
                1 => (c.pixel_format, raw[index.to_usize()]) = self.pixel_format(BLOCK, item),
                2 => c.blending_enabled = self.flag(BLOCK, item),
                3 => c.source_rgb_blend_factor = self.metal(BLOCK, item, "blend factor"),
                4 => c.destination_rgb_blend_factor = self.metal(BLOCK, item, "blend factor"),
                5 => c.rgb_blend_operation = self.metal(BLOCK, item, "blend operation"),
                6 => c.source_alpha_blend_factor = self.metal(BLOCK, item, "blend factor"),
                7 => c.destination_alpha_blend_factor = self.metal(BLOCK, item, "blend factor"),
                8 => c.alpha_blend_operation = self.metal(BLOCK, item, "blend operation"),
                9 => c.write_mask = self.metal(BLOCK, item, "colour write mask"),
                _ => self.ignore(ignored, BLOCK, item),
            }
        });
    }
}

/// A vertex attribute as Metal's array holds one the blob does not set: format `Invalid`, offset
/// 0, buffer 0 (measured).
fn unset_vertex_attribute() -> VertexAttributeDescriptor {
    VertexAttributeDescriptor::new(VertexFormat::Invalid, 0, 0)
}

/// A vertex layout as Metal's array holds one the blob does not set: stride 0, per vertex, rate 1
/// (measured).
fn unset_vertex_layout() -> VertexBufferLayoutDescriptor {
    VertexBufferLayoutDescriptor::new(Some(0))
}

/// A layout's stride: `None` for `MTLBufferLayoutStrideDynamic`.
fn stride(raw: u32) -> Option<u32> {
    (raw != DYNAMIC_STRIDE).then_some(raw)
}

/// A stage-input attribute as Metal's array holds one the blob does not set: format `Invalid`,
/// offset 0, buffer 0 (measured).
fn unset_stage_input_attribute() -> StageInputAttributeDescriptor {
    StageInputAttributeDescriptor::new(AttributeFormat::Invalid, 0, 0)
}

/// A stage-input layout as Metal's array holds one the blob does not set: stride 0, per vertex,
/// rate 1 (measured).
fn unset_stage_input_layout() -> StageInputLayoutDescriptor {
    StageInputLayoutDescriptor::new(Some(0))
}

/// The compute form: `None` for an empty blob, as Metal answers nil for one.
///
/// | key | property | the translator's field |
/// |---|---|---|
/// | 0 | `computeFunction` (ref) | `kernel`; a ref that resolves to nil aborts |
/// | 1 | `threadGroupSizeIsMultipleOfThreadExecutionWidth` | `threadgroup_size_is_multiple_of_execution_width` |
/// | 2 | `label` (string) | `label`, the state's (`PipelineOptions::label`) |
/// | 3 | `stageInputDescriptor` (block) | `stage_input` |
/// | 6 | `buffers` mutability (array) | `buffers` |
/// | 7 | `supportIndirectCommandBuffers` | `support_indirect_command_buffers` |
/// | 8 | `maxTotalThreadsPerThreadgroup` | `max_total_threads_per_threadgroup` (0 is `None`) |
/// | 10 | `textureWriteRoundingMode` | none: a kernel writes alike at every mode the device supports |
/// | 12 | [`COMPUTE_SPI`] | none: dies but at 0 |
pub fn compute(
    hold: Hold<'_>,
    data: &[u8],
    resolve: FunctionResolver<'_, VulkanPlatform>,
) -> RaiseResult<Option<Decoded<ComputePipelineDescriptor>>> {
    const BLOCK: &str = "compute";
    if data.is_empty() {
        return Ok(None);
    }
    let r = Reader::new(COMPUTE_SEL, data);
    let mut ignored = Vec::new();
    let mut label = None;
    let mut kernel = None;
    let mut stage_input = None;
    let mut buffers = BTreeMap::new();
    let (mut multiple_of_width, mut icb, mut max_threads) = (false, false, 0);
    let mut spi = BTreeMap::new();
    for e in r.entries(0, BLOCK) {
        match e.key {
            0 => {
                let function = r.function(hold, BLOCK, e, &mut *resolve)?;
                kernel = Some(function.unwrap_or_else(|| {
                    metal_aborts(COMPUTE_SEL, "a compute function ref that resolves to nil")
                }));
            }
            1 => multiple_of_width = r.flag(BLOCK, e),
            2 => label = Some(r.label(0, BLOCK, e)),
            3 => {
                stage_input = r
                    .nested(0, BLOCK, e)
                    .map(|at| r.stage_input(at, &mut ignored));
            }
            6 => r.buffers(0, "compute.buffers", e, &mut buffers, &mut ignored),
            7 => icb = r.flag(BLOCK, e),
            8 => max_threads = r.value(BLOCK, e),
            10 => {
                r.rounding_mode(BLOCK, e);
            }
            _ => r.other(BLOCK, e, (COMPUTE_SPI, &mut spi), &mut ignored),
        }
    }
    let kernel = kernel.unwrap_or_else(|| {
        metal_aborts(
            COMPUTE_SEL,
            "a compute descriptor with no function, at the state's creation",
        )
    });
    decide_spi((COMPUTE_SEL, BLOCK), COMPUTE_SPI, &spi, |_, _| None);
    let mut desc = ComputePipelineDescriptor::new(kernel);
    desc.stage_input = stage_input;
    desc.max_total_threads_per_threadgroup = (max_threads != 0).then_some(max_threads);
    desc.threadgroup_size_is_multiple_of_execution_width = multiple_of_width;
    desc.support_indirect_command_buffers = icb;
    desc.buffers = buffers;
    Ok(Some(Decoded {
        desc,
        ignored,
        nearest_even: false,
        label,
        raw_format: None,
    }))
}

/// The render form: `None` for an empty blob, as Metal answers nil for one.
///
/// | key | property | the translator's field |
/// |---|---|---|
/// | 0 | `label` (string) | `label`, the state's (`PipelineOptions::label`) |
/// | 1 | `vertexFunction` (ref) | `vertex`; none at the end aborts |
/// | 2 | `fragmentFunction` (ref) | `fragment`, which the runtime checks as Metal links it under rasterization off too |
/// | 3 | `vertexDescriptor` (block) | `vertex_descriptor` |
/// | 4 | `rasterSampleCount` | `raster_sample_count` |
/// | 5 | `alphaToCoverageEnabled` | `alpha_to_coverage` |
/// | 6 | `alphaToOneEnabled` | `alpha_to_one` |
/// | 7 | `rasterizationEnabled` | `rasterization_enabled` |
/// | 8 | `colorAttachments` (array) | `color_attachments` |
/// | 9 | `depthAttachmentPixelFormat` | `depth_format` |
/// | 10 | `stencilAttachmentPixelFormat` | `stencil_format` |
/// | 11 | `inputPrimitiveTopology` | `input_primitive_topology` |
/// | 12 | `tessellationPartitionMode` | `tessellation`'s `partition_mode` |
/// | 13 | `maxTessellationFactor` | `tessellation`'s `max_tessellation_factor` |
/// | 14 | `tessellationFactorScaleEnabled` | `tessellation`'s `factor_scale_enabled` |
/// | 15 | `tessellationFactorFormat` | `tessellation`'s `factor_format` |
/// | 16 | `tessellationControlPointIndexType` | `tessellation`'s `control_point_index_type` |
/// | 17 | `tessellationFactorStepFunction` | `tessellation`'s `factor_step_function` |
/// | 18 | `tessellationOutputWindingOrder` | `tessellation`'s `output_winding_order` |
/// | 21 | `vertexBuffers` mutability (array) | `vertex_buffers` |
/// | 22 | `fragmentBuffers` mutability (array) | `fragment_buffers` |
/// | 25 | `supportIndirectCommandBuffers` | `support_indirect_command_buffers` |
/// | 26 | `maxVertexAmplificationCount` | `max_vertex_amplification_count` |
/// | 30 | `textureWriteRoundingMode` | `texture_write_rounding_mode`, and `nearest_even` at [`NEAREST_EVEN`] (the module docs) |
/// | 24, 27, 28, 45 to 58 | [`RENDER_SPI`] | none: decided once the block is read ([`render_spi`]) |
pub fn render(
    hold: Hold<'_>,
    data: &[u8],
    resolve: FunctionResolver<'_, VulkanPlatform>,
) -> RaiseResult<Option<Decoded<RenderPipelineDescriptor>>> {
    const BLOCK: &str = "render";
    log::debug!("{}", Blob(RENDER_SEL, data));
    if data.is_empty() {
        return Ok(None);
    }
    let r = Reader::new(RENDER_SEL, data);
    let mut ignored = Vec::new();
    let mut label = None;
    let (mut vertex, mut fragment) = (None, None);
    let mut vertex_descriptor = None;
    let mut colors = [ColorAttachmentDescriptor::new(PixelFormat::Invalid); 8];
    let mut raw_colors = [None; 8];
    let (mut depth, mut stencil) = ((PixelFormat::Invalid, None), (PixelFormat::Invalid, None));
    let (mut samples, mut amplification) = (1, 1);
    let (mut alpha_to_coverage, mut alpha_to_one, mut rasterization) = (false, false, true);
    let mut topology = PrimitiveTopologyClass::Unspecified;
    let mut tessellation = TessellationDescriptor::default();
    let (mut vertex_buffers, mut fragment_buffers) = (BTreeMap::new(), BTreeMap::new());
    let mut icb = false;
    let mut rounding = 0;
    let mut spi = BTreeMap::new();
    for e in r.entries(0, BLOCK) {
        match e.key {
            0 => label = Some(r.label(0, BLOCK, e)),
            1 => vertex = r.function(hold, BLOCK, e, &mut *resolve)?,
            2 => fragment = r.function(hold, BLOCK, e, &mut *resolve)?,
            3 => {
                vertex_descriptor = r
                    .nested(0, BLOCK, e)
                    .map(|at| r.vertex_descriptor(at, &mut ignored));
            }
            4 => samples = r.value(BLOCK, e),
            5 => alpha_to_coverage = r.flag(BLOCK, e),
            6 => alpha_to_one = r.flag(BLOCK, e),
            7 => rasterization = r.flag(BLOCK, e),
            8 => r.color_attachments(e, (&mut colors, &mut raw_colors), &mut ignored),
            9 => depth = r.pixel_format(BLOCK, e),
            10 => stencil = r.pixel_format(BLOCK, e),
            11 => topology = r.metal(BLOCK, e, "primitive topology class"),
            12 => tessellation.partition_mode = r.metal(BLOCK, e, "tessellation partition mode"),
            13 => tessellation.max_tessellation_factor = r.value(BLOCK, e),
            14 => tessellation.factor_scale_enabled = r.flag(BLOCK, e),
            15 => tessellation.factor_format = r.metal(BLOCK, e, "tessellation factor format"),
            16 => {
                tessellation.control_point_index_type =
                    r.metal(BLOCK, e, "tessellation control point index type");
            }
            17 => {
                tessellation.factor_step_function =
                    r.metal(BLOCK, e, "tessellation factor step function");
            }
            18 => tessellation.output_winding_order = r.metal(BLOCK, e, "winding"),
            21 => r.buffers(
                0,
                "render.vertexBuffers",
                e,
                &mut vertex_buffers,
                &mut ignored,
            ),
            22 => r.buffers(
                0,
                "render.fragmentBuffers",
                e,
                &mut fragment_buffers,
                &mut ignored,
            ),
            25 => icb = r.flag(BLOCK, e),
            26 => amplification = r.value(BLOCK, e),
            30 => rounding = r.rounding_mode(BLOCK, e),
            _ => r.other(BLOCK, e, (RENDER_SPI, &mut spi), &mut ignored),
        }
    }
    let vertex = vertex.unwrap_or_else(|| {
        metal_aborts(
            RENDER_SEL,
            "a render descriptor with no vertex function, at the state's creation",
        )
    });
    render_spi(&spi, samples, &colors, &raw_colors);
    let raw_format = RawFormat::first(&raw_colors, depth.1, stencil.1);
    let (depth, stencil) = (depth.0, stencil.0);
    let format = |f: PixelFormat| (f != PixelFormat::Invalid).then_some(f);
    let mut desc = RenderPipelineDescriptor::new(vertex);
    desc.fragment = fragment;
    desc.vertex_descriptor = vertex_descriptor;
    desc.color_attachments = colors.map(|c| format(c.pixel_format).map(|_| c));
    desc.depth_format = format(depth);
    desc.stencil_format = format(stencil);
    desc.raster_sample_count = samples;
    desc.alpha_to_coverage = alpha_to_coverage;
    desc.alpha_to_one = alpha_to_one;
    desc.rasterization_enabled = rasterization;
    desc.input_primitive_topology = topology;
    desc.max_vertex_amplification_count = amplification;
    desc.tessellation = (tessellation != TessellationDescriptor::default()).then_some(tessellation);
    desc.support_indirect_command_buffers = icb;
    desc.vertex_buffers = vertex_buffers;
    desc.fragment_buffers = fragment_buffers;
    desc.texture_write_rounding_mode = texture_write_rounding(rounding);
    let nearest_even = rounding == NEAREST_EVEN;
    Ok(Some(Decoded {
        desc,
        ignored,
        nearest_even,
        label,
        raw_format,
    }))
}

/// The tile form: `None` for an empty blob, as Metal answers nil for one. The blob is the record's
/// `descriptorLength` bytes, which Metal's data spans.
///
/// | key | property | the translator's field |
/// |---|---|---|
/// | 0 | `label` (string) | `label`, the state's (`PipelineOptions::label`) |
/// | 1 | `tileFunction` (ref) | `tile`; none at the end aborts |
/// | 2 | `rasterSampleCount` | `raster_sample_count` |
/// | 3 | `colorAttachments` (array of `{0 index, 1 pixelFormat}`) | `color_attachment_formats` |
/// | 4 | `threadgroupSizeMatchesTileSize` | `threadgroup_size_matches_tile_size` |
/// | 5 | `tileBuffers` mutability (array) | `buffers` |
/// | 6 | `maxTotalThreadsPerThreadgroup` | `max_total_threads_per_threadgroup` (0 is `None`) |
/// | 8 | `textureWriteRoundingMode` | none: a tile function writes alike at every mode the device supports |
pub fn tile(
    hold: Hold<'_>,
    data: &[u8],
    resolve: FunctionResolver<'_, VulkanPlatform>,
) -> RaiseResult<Option<Decoded<TileRenderPipelineDescriptor>>> {
    const BLOCK: &str = "tile";
    const COLORS: &str = "tile.colorAttachments";
    if data.is_empty() {
        return Ok(None);
    }
    let r = Reader::new(TILE_SEL, data);
    let mut ignored = Vec::new();
    let mut label = None;
    let mut function = None;
    let mut formats = [PixelFormat::Invalid; 8];
    let mut raw_formats = [None; 8];
    let (mut samples, mut matches_tile, mut max_threads) = (1, false, 0);
    let mut buffers = BTreeMap::new();
    for e in r.entries(0, BLOCK) {
        match e.key {
            0 => label = Some(r.label(0, BLOCK, e)),
            1 => function = r.function(hold, BLOCK, e, &mut *resolve)?,
            2 => samples = r.value(BLOCK, e),
            3 => r.elements(0, COLORS, e, LAST_COLOR_INDEX, |index, item| {
                match item.key {
                    0 => {}
                    1 => {
                        let raw = u64::from(r.value(COLORS, item));
                        if !crate::gpu::format_info::setter_takes(TILE_SEL, raw) {
                            metal_aborts(
                                TILE_SEL,
                                &format!("pixel format {raw}, not a valid MTLPixelFormat"),
                            );
                        }
                        let i = index.to_usize();
                        (formats[i], raw_formats[i]) = r.pixel_format(COLORS, item);
                    }
                    _ => r.ignore(&mut ignored, COLORS, item),
                }
            }),
            4 => matches_tile = r.flag(BLOCK, e),
            5 => r.buffers(0, "tile.tileBuffers", e, &mut buffers, &mut ignored),
            6 => max_threads = r.value(BLOCK, e),
            8 => {
                r.rounding_mode(BLOCK, e);
            }
            _ => r.ignore(&mut ignored, BLOCK, e),
        }
    }
    let function = function.unwrap_or_else(|| {
        metal_aborts(
            TILE_SEL,
            "a tile descriptor with no tile function, at the state's creation",
        )
    });
    let mut desc = TileRenderPipelineDescriptor::new(function);
    desc.color_attachment_formats = formats.map(|f| (f != PixelFormat::Invalid).then_some(f));
    desc.raster_sample_count = samples;
    desc.threadgroup_size_matches_tile_size = matches_tile;
    desc.max_total_threads_per_threadgroup = (max_threads != 0).then_some(max_threads);
    desc.buffers = buffers;
    Ok(Some(Decoded {
        desc,
        ignored,
        nearest_even: false,
        label,
        raw_format: RawFormat::first(&raw_formats, None, None),
    }))
}

#[cfg(test)]
#[cfg(target_os = "macos")]
mod draws;

#[cfg(test)]
pub(crate) mod tests {
    use metal2vulkan::metal::StepFunction;
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::raise::{OnFault, Raise};
    use reims_vgpu_core::testkit::capture::{catch_death, raise};

    use super::*;
    use crate::gpu::device::tests::dev;

    /// A format refusal that names an attachment dies named by that attachment's format, and one
    /// that names none dies naming no attachment, its line the refusal and the descriptor's
    /// formats; another refusal, or a format refusal of a state that holds no format, is the
    /// caller's
    #[test]
    fn a_format_refusal_blames_only_the_attachment_it_names() {
        use metal2vulkan::error::{Attachment, Feature, Location, Unsupported};
        let sel = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
        let format = |at: Option<Attachment>| {
            let u = Unsupported::not_implemented(Feature::Format);
            let u = match at {
                Some(a) => u.at(Location::render_pipeline().with_attachment(a)),
                None => u,
            };
            rt::Error::Unsupported(u)
        };
        let colors = [
            Some(PixelFormat::BGRA8Unorm),
            Some(PixelFormat::RGBA16Float),
        ];
        let refused = |e: rt::Error| {
            *crate::answer::LAST_LINE.lock() = None;
            let listed = || "the descriptor's formats: listed".to_owned();
            let site = catch_death(|| {
                RawFormat::refused(sel, &colors, (None, None), listed, e);
            })
            .unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
            crate::answer::LAST_LINE.lock().take().unwrap_or_default()
        };
        let line = refused(format(Some(Attachment::Color(1))));
        assert!(
            line.contains("a state whose colour attachment 1 is pixel format 115"),
            "{line}"
        );
        let line = refused(format(None));
        assert!(!line.contains("colour attachment"), "{line}");
        assert!(
            line.contains("at an attachment it does not name or in a function")
                && line.ends_with("; the descriptor's formats: listed"),
            "{line}"
        );
        let other = rt::Error::NotImplemented(rt::RuntimeGap::Textures);
        let back = RawFormat::refused(sel, &colors, (None, None), String::new, other);
        assert!(matches!(back, rt::Error::NotImplemented(_)));
        let back = RawFormat::refused(sel, &[None], (None, None), String::new, format(None));
        assert!(matches!(back, rt::Error::Unsupported(_)));
    }

    /// The hex of `s`, spaces allowed.
    pub(crate) fn hex(s: &str) -> Vec<u8> {
        let s: String = s.split_whitespace().collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// A render blob's debug line is its selector, its length, then every byte in lowercase hex,
    /// which [`hex`] reads back as the blob; an empty blob's has no bytes.
    #[test]
    fn a_blobs_debug_line_is_its_bytes_in_hex() {
        let data = [0x02, 0x00, 0x04, 0xff, 0x0a, 0x10, 0x00];
        let line = Blob(RENDER_SEL, &data).to_string();
        assert_eq!(
            line,
            "newRenderPipelineDescriptorWithSerializedData:deserializationContext: 7 bytes: \
             020004ff0a1000"
        );
        assert_eq!(hex(line.rsplit(' ').next().unwrap()), data);
        assert_eq!(
            Blob(RENDER_SEL, &[]).to_string(),
            format!("{RENDER_SEL} 0 bytes: ")
        );
    }

    /// The one function of a fixture metallib.
    pub(crate) fn fixture_function(bytes: &[u8]) -> VulkanFunction {
        let library = dev().new_library(bytes).unwrap().unwrap().unwrap();
        let name = library.function_names().remove(0);
        library
            .new_function(OnFault::Return, &name)
            .unwrap()
            .unwrap()
    }

    pub(crate) const KERNEL: &[u8] =
        include_bytes!("../../../../tests/suite/fixtures/shaders/kernel.metallib");
    pub(crate) const STAGEIN: &[u8] =
        include_bytes!("../../../../tests/suite/fixtures/shaders/compute_stagein.metallib");

    /// A resolver answering `f` for every ref, and the refs it was asked for.
    pub(crate) struct Resolve {
        pub(crate) f: Option<VulkanFunction>,
        pub(crate) refs: Vec<u64>,
        /// Refs it raises for instead.
        pub(crate) raises: Vec<u64>,
        /// Refs it resolves to nil instead.
        pub(crate) nil: Vec<u64>,
    }

    impl Resolve {
        pub(crate) fn answering(f: Option<VulkanFunction>) -> Resolve {
            Resolve {
                f,
                refs: Vec::new(),
                raises: Vec::new(),
                nil: Vec::new(),
            }
        }

        pub(crate) fn call(
            &mut self,
            _hold: Hold<'_>,
            r: u64,
        ) -> RaiseResult<Option<VulkanFunction>> {
            use reims_vgpu_core::platform::Handle as _;
            self.refs.push(r);
            if self.raises.contains(&r) {
                return Err(raise(OnFault::Return, "resolver", "lookup"));
            }
            if self.nil.contains(&r) {
                return Ok(None);
            }
            Ok(self.f.as_ref().map(|f| f.clone_ref()))
        }
    }

    fn decode(
        blob: &[u8],
        resolve: &mut Resolve,
    ) -> RaiseResult<Option<Decoded<ComputePipelineDescriptor>>> {
        compute(Hold::NONE, blob, &mut |h, r| resolve.call(h, r))
    }

    /// The decoded descriptor of `blob`, its function `f`.
    fn decoded(blob: &[u8], f: &VulkanFunction) -> (Decoded<ComputePipelineDescriptor>, Vec<u64>) {
        use reims_vgpu_core::platform::Handle as _;
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        let d = decode(blob, &mut resolve).unwrap().unwrap();
        (d, resolve.refs)
    }

    /// `compute_objects.py`'s `compute_blob(fn_ref, icb, max_threads)`.
    fn compute_blob(fn_ref: u32, icb: bool, max_threads: Option<u32>) -> Vec<u8> {
        let mut entries = Vec::new();
        if icb {
            entries.push((7u8, 1u32));
        }
        if let Some(n) = max_threads {
            entries.push((8, n));
        }
        entries.push((0, fn_ref));
        let mut blob = vec![u8::try_from(entries.len()).unwrap()];
        for (k, v) in entries {
            blob.extend([k, 4]);
            blob.extend(v.to_le_bytes());
        }
        blob
    }

    /// `compute_objects.py`'s `stage_in_blob(fn_ref)`: attribute 0 `UInt` (36) from buffer 1,
    /// layout 1 stride 4, step function `ThreadPositionInGridX` (5).
    pub(crate) fn stage_in_blob(fn_ref: u32) -> Vec<u8> {
        let mut blob = hex(
            "02 03 04 10 00 00 00 00 04 00 00 00 00 00 00 00 02 00 04 10 00 00 00 01 04 2c 00 00 \
             00 00 00 00 01 00 00 00 08 00 00 00 03 00 04 00 00 00 00 01 04 24 00 00 00 03 04 01 \
             00 00 00 00 01 00 00 00 08 00 00 00 03 00 04 01 00 00 00 01 04 05 00 00 00 03 04 04 \
             00 00 00",
        );
        blob[9..13].copy_from_slice(&fn_ref.to_le_bytes());
        blob
    }

    /// The suite's compute blobs decode key by key, each function resolved by its ref
    #[test]
    fn compute_blobs_decode_key_by_key() {
        let f = fixture_function(KERNEL);
        let kernel = StageSource::Function(f.function_descriptor());
        let (d, refs) = decoded(&compute_blob(0x2a, false, None), &f);
        assert_eq!(refs, vec![0x2a]);
        assert_eq!(d.desc, ComputePipelineDescriptor::new(kernel.clone()));
        assert!(d.ignored.is_empty());
        let (d, _) = decoded(&compute_blob(7, true, Some(64)), &f);
        assert!(d.desc.support_indirect_command_buffers);
        assert_eq!(d.desc.max_total_threads_per_threadgroup, Some(64));
        assert!(!d.desc.threadgroup_size_is_multiple_of_execution_width);
        // testing.md 6.18's: + threadGroupSizeIsMultipleOfThreadExecutionWidth
        let (d, _) = decoded(&hex("02 01 04 01000000 00 04 07000000"), &f);
        assert!(d.desc.threadgroup_size_is_multiple_of_execution_width);
        // maxTotalThreadsPerThreadgroup 0 is Metal's default, the device's maximum
        let (d, _) = decoded(&compute_blob(7, false, Some(0)), &f);
        assert_eq!(d.desc.max_total_threads_per_threadgroup, None);
    }

    /// The stage-input blob: attribute 0 from buffer 1, layout 1 stepping by grid X at stride 4,
    /// the index type and buffer Metal's defaults
    #[test]
    fn the_stage_input_blob_decodes_its_attributes_and_layouts() {
        let f = fixture_function(STAGEIN);
        let (d, refs) = decoded(&stage_in_blob(9), &f);
        assert_eq!(refs, vec![9]);
        let s = d.desc.stage_input.unwrap();
        assert_eq!(
            s.attributes,
            BTreeMap::from([(
                0,
                StageInputAttributeDescriptor::new(AttributeFormat::UInt, 0, 1)
            )])
        );
        let mut layout = StageInputLayoutDescriptor::new(Some(4));
        layout.step_function = StepFunction::ThreadPositionInGridX;
        assert_eq!(s.layouts, BTreeMap::from([(1, layout)]));
        assert_eq!((s.index_type, s.index_buffer_index), (IndexType::UInt16, 0));
    }

    /// Buffers' mutabilities, a label, a dynamic stride, an attribute whose buffer the blob gives
    /// no layout (Metal's default one), as serialized on this host by
    /// `-serializeComputePipelineDescriptor:`
    #[test]
    fn buffers_labels_and_default_layouts_decode_as_metal_holds_them() {
        let f = fixture_function(KERNEL);
        // buffers[3] Immutable, buffers[0] Mutable
        let blob = hex(
            "02 06 04 10000000 00 04 11000000 000000 02000000 0c000000 19000000 \
             02 00 04 00000000 01 04 01000000 02 00 04 03000000 01 04 02000000",
        );
        let (d, _) = decoded(&blob, &f);
        assert_eq!(
            d.desc.buffers,
            BTreeMap::from([(0, Mutability::Mutable), (3, Mutability::Immutable)])
        );
        assert_eq!(d.label, None);
        // label "ab", then the empty label
        let (d, _) = decoded(&hex("02 02 04 0d000000 00 04 11000000 616200"), &f);
        assert_eq!(d.label.as_deref(), Some("ab"));
        let (d, _) = decoded(&hex("02 02 04 0d000000 00 04 11000000 00"), &f);
        assert_eq!(d.label.as_deref(), Some(""));
        // attribute 0 UInt from buffer 1, layout 1 dynamic and by grid X
        let blob = hex(
            "02 03 04 10000000 00 04 11000000 000000 02 00 04 10000000 01 04 2c000000 000000 \
             01000000 08000000 03 00 04 00000000 01 04 24000000 03 04 01000000 00 \
             01000000 08000000 03 00 04 01000000 01 04 05000000 03 04 ffffffff",
        );
        let (d, _) = decoded(&blob, &f);
        let s = d.desc.stage_input.unwrap();
        assert_eq!(s.layouts[&1].stride, None);
        // attribute 2 from buffer 5, with no layout: Metal's default for buffer 5
        let blob = hex(
            "02 03 04 10000000 00 04 11000000 000000 01 00 04 08000000 00 \
             01000000 08000000 02 00 04 02000000 03 04 05000000",
        );
        let (d, _) = decoded(&blob, &f);
        let s = d.desc.stage_input.unwrap();
        assert_eq!(
            s.attributes[&2],
            StageInputAttributeDescriptor::new(AttributeFormat::Invalid, 0, 5)
        );
        assert_eq!(s.layouts, BTreeMap::from([(5, unset_stage_input_layout())]));
        assert_eq!(s.layouts[&5].step_function, StepFunction::PerVertex);
    }

    /// An element's entries set the element its key 0 selected last; one before any index sets
    /// nothing; a repeated top-level key's last value wins, its function resolved each time; a
    /// boolean is any value but 0; an offset of 0 is nil
    #[test]
    fn entries_apply_in_order() {
        let f = fixture_function(KERNEL);
        let element = |entries: &[(u8, u32)]| {
            let mut b = vec![u8::try_from(entries.len()).unwrap()];
            for &(k, v) in entries {
                b.extend([k, 4]);
                b.extend(v.to_le_bytes());
            }
            b
        };
        let buffers = |entries: &[(u8, u32)]| {
            let mut blob = hex("02 06 04 10000000 00 04 11000000 000000");
            blob.extend(1u32.to_le_bytes());
            blob.extend(8u32.to_le_bytes());
            blob.extend(element(entries));
            blob
        };
        let (d, _) = decoded(&buffers(&[(1, 2), (0, 3)]), &f);
        assert!(d.desc.buffers.is_empty());
        let (d, _) = decoded(&buffers(&[(0, 3), (1, 2), (0, 4)]), &f);
        assert_eq!(d.desc.buffers, BTreeMap::from([(3, Mutability::Immutable)]));
        let (d, refs) = decoded(
            &hex("03 08 04 40000000 00 04 05000000 08 04 80000000 00 04 06000000"),
            &f,
        );
        assert_eq!(refs, vec![5]);
        assert_eq!(d.desc.max_total_threads_per_threadgroup, Some(128));
        let (_, refs) = decoded(&hex("02 00 04 05000000 00 04 06000000"), &f);
        assert_eq!(refs, vec![5, 6]);
        for raw in [2u32, 3, 0x100, 0x101] {
            let mut blob = hex("02 01 04 00000000 00 04 11000000");
            blob[3..7].copy_from_slice(&raw.to_le_bytes());
            assert!(
                decoded(&blob, &f)
                    .0
                    .desc
                    .threadgroup_size_is_multiple_of_execution_width
            );
        }
        let (d, _) = decoded(&hex("03 03 04 00000000 06 04 00000000 00 04 11000000"), &f);
        assert_eq!((d.desc.stage_input, d.desc.buffers.len()), (None, 0));
        // An array element at offset 0 holds nothing
        let mut blob = hex("02 06 04 0d000000 00 04 11000000");
        blob.extend([1, 0, 0, 0, 0, 0, 0, 0]);
        assert!(decoded(&blob, &f).0.desc.buffers.is_empty());
    }

    /// Keys no block defines are kept as the blob had them, as Metal ignores them, whatever their
    /// value's size; trailing bytes are ignored
    #[test]
    fn undefined_keys_are_kept_as_metal_ignores_them() {
        let f = fixture_function(KERNEL);
        let (d, _) = decoded(
            &hex("03 04 04 05000000 ff 04 ffffffff 00 04 11000000 abababab"),
            &f,
        );
        assert_eq!(
            d.ignored,
            vec![
                Ignored {
                    block: "compute",
                    key: 4,
                    value: vec![5, 0, 0, 0]
                },
                Ignored {
                    block: "compute",
                    key: 0xff,
                    value: vec![0xff; 4]
                },
            ]
        );
        assert_eq!(
            format!("{:?}", d.ignored[0]),
            "compute.field_04 = [05, 00, 00, 00]"
        );
        // A last entry's ignored value past the blob's end is never read (measured: Metal takes
        // it)
        let (d, _) = decoded(&hex("02 00 04 11000000 05 11 0102"), &f);
        assert_eq!(d.ignored[0].value, vec![1, 2]);
        // A stage-input block's, an attribute's and a buffer's undefined keys
        let blob = hex(
            "02 03 04 10000000 00 04 11000000 000000 02 00 04 10000000 09 04 07000000 000000 \
             01000000 08000000 02 00 04 00000000 05 04 06000000",
        );
        let (d, _) = decoded(&blob, &f);
        let keys: Vec<_> = d.ignored.iter().map(|i| (i.block, i.key)).collect();
        assert_eq!(
            keys,
            vec![
                ("compute.stageInputDescriptor.attributes", 5),
                ("compute.stageInputDescriptor", 9)
            ]
        );
    }

    /// Array elements that all lie at one block of undefined keys keep only the first
    /// [`KEPT_IGNORED`] of its entries, in the blob's order, however many elements reach it, as
    /// Metal's deserializer allocates nothing for them; the elements' defined keys still decode
    #[test]
    fn ignored_entries_stay_bounded_however_many_elements_share_a_block() {
        const ELEMENTS: u32 = 256;
        const KEYS: u8 = 200;
        let f = fixture_function(KERNEL);
        // buffers (key 6) at 13, then the function's ref
        let mut blob = hex("02 06 04 0d000000 00 04 11000000");
        // The array at 13: every element at the one block after the offsets.
        blob.extend(ELEMENTS.to_le_bytes());
        let block = 4 + 4 * ELEMENTS;
        for _ in 0..ELEMENTS {
            blob.extend(block.to_le_bytes());
        }
        // The block: index 3, mutability Immutable, then KEYS entries of key 5, which a buffer
        // element does not define, of 255 bytes each.
        blob.push(KEYS + 2);
        blob.extend(hex("00 04 03000000 01 04 02000000"));
        for k in 0..KEYS {
            blob.extend([5, 255]);
            blob.extend([k; 255]);
        }
        let (d, _) = decoded(&blob, &f);
        assert_eq!(d.desc.buffers, BTreeMap::from([(3, Mutability::Immutable)]));
        assert_eq!(d.ignored.len(), KEPT_IGNORED);
        for (i, ignored) in d.ignored.iter().enumerate() {
            let k = u8::try_from(i).unwrap();
            assert_eq!((ignored.block, ignored.key), ("compute.buffers", 5));
            assert_eq!(ignored.value, vec![k; 255], "{i}");
        }
    }

    /// An array's offsets lie in the blob before anything is sized from its count: a count past
    /// them, up to 0xffffffff, dies at the selector, and one they hold reads each element
    #[test]
    fn an_array_count_past_the_blob_dies_before_it_sizes_anything() {
        for count in [2u32, 0x1000_0000, u32::MAX] {
            let mut blob = count.to_le_bytes().to_vec();
            blob.extend(hex("08000000 00"));
            let site = catch_death(|| Reader::new("s", &blob).array(0, "b")).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, "s"),
                "{count:#x}"
            );
        }
        let blob = hex("02000000 0c000000 00000000 00");
        assert_eq!(Reader::new("s", &blob).array(0, "b"), vec![Some(12), None]);
    }

    /// The decode's death site and its detail
    fn dies(blob: &[u8], resolve: &mut Resolve) -> reims_vgpu_core::raise::death::DeathSite {
        catch_death(|| decode(blob, resolve).is_ok()).unwrap_err()
    }

    /// Every blob Metal aborts on dies at the deserializer's selector: the suite's garbage and
    /// truncated blobs, offsets, counts and labels past the end, a label at 0, an index past
    /// Metal's arrays, and a function ref that resolves to nil (after its resolve, before the
    /// next)
    #[test]
    fn blobs_metal_aborts_on_die_at_the_selector() {
        let f = fixture_function(KERNEL);
        let blob = compute_blob(5, false, None);
        let mut cases = vec![
            ("compute_death_blob_garbage", vec![0xab; 64]),
            ("compute_death_blob_truncated1", blob[..1].to_vec()),
            ("compute_death_blob_truncated5", blob[..5].to_vec()),
        ];
        for (what, h) in [
            (
                "a label with no NUL",
                "02 02 04 0d000000 00 04 11000000 6162",
            ),
            ("a label at the end", "02 02 04 0d000000 00 04 11000000"),
            ("a label at 0", "02 02 04 00000000 00 04 11000000"),
            (
                "a label far past the end",
                "02 02 04 00100000 00 04 11000000",
            ),
            (
                "a stage input past the end",
                "02 03 04 0d000000 00 04 11000000",
            ),
            (
                "an array count past the end",
                "02 06 04 0d000000 00 04 11000000 e8030000",
            ),
            (
                "an array count of 0xffffffff",
                "02 06 04 0d000000 00 04 11000000 ffffffff 08000000",
            ),
            (
                "an element past the end",
                "02 06 04 0d000000 00 04 11000000 01000000 00100000",
            ),
            (
                "an element at the end",
                "02 06 04 0d000000 00 04 11000000 01000000 08000000",
            ),
            (
                "a value past the end",
                "03 08 04 40000000 08 04 80000000 00 04 1100",
            ),
            ("a value too long", "02 08 ff 40000000 00 04 11000000"),
        ] {
            cases.push((what, hex(h)));
        }
        let buffer = |index: u32| {
            let mut b = hex("02 06 04 10000000 00 04 11000000 000000 01000000 08000000 02 00 04");
            b.extend(index.to_le_bytes());
            b.extend(hex("01 04 02000000"));
            b
        };
        cases.push(("buffer index 31", buffer(31)));
        for (what, blob) in cases {
            use reims_vgpu_core::platform::Handle as _;
            let site = dies(&blob, &mut Resolve::answering(Some(f.clone_ref())));
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, COMPUTE_SEL),
                "{what}"
            );
        }
        decoded(&buffer(30), &f);
        // The ref resolves, then the decode dies before the next entry
        let mut nil = Resolve::answering(None);
        let site = dies(&hex("02 00 04 99000000 00 04 98000000"), &mut nil);
        assert_eq!(site.file, COMPUTE_SEL);
        assert_eq!(nil.refs, vec![0x99]);
        // A function resolved before an entry past the end
        use reims_vgpu_core::platform::Handle as _;
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        dies(&hex("02 00 04 11000000 08"), &mut resolve);
        assert_eq!(resolve.refs, vec![0x11]);
        // No function entry: Metal aborts at the state's creation
        let site = dies(&hex("00"), &mut Resolve::answering(None));
        assert_eq!(site.file, COMPUTE_SEL);
    }

    /// An empty blob is nil, and nothing is resolved
    #[test]
    fn an_empty_blob_is_nil() {
        let mut resolve = Resolve::answering(None);
        assert!(decode(&[], &mut resolve).unwrap().is_none());
        assert!(resolve.refs.is_empty());
    }

    /// The resolver's raise ends the decode and is the caller's; no later ref is resolved
    #[test]
    fn a_resolver_raise_is_the_callers() {
        let f = fixture_function(KERNEL);
        use reims_vgpu_core::platform::Handle as _;
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        resolve.raises.push(5);
        let r: Raise = decode(&hex("03 00 04 05000000 00 04 06000000 08"), &mut resolve)
            .err()
            .unwrap();
        assert_eq!(r.detail(), Some("lookup"));
        assert_eq!(resolve.refs, vec![5]);
    }

    /// `render_lib.py`'s blobs, as `render_blobs.m` serializes them, with its sentinel refs: vertex
    /// `a1a1a1a1`, fragment `b2b2b2b2`, tile `c3c3c3c3`.
    pub(crate) const BLOBS: &[(&str, &str)] = &[
        (
            "basic",
            "030804140000000104a1a1a1a10204b2b2b2b200010000000800000002000400000000010446000000",
        ),
        (
            "blend",
            "030804140000000104a1a1a1a10204b2b2b2b20001000000080000000600040000000001044600000002040100\
             000003040b00000004040100000006040d000000",
        ),
        (
            "vdesc",
            "0403041c0000000804800000000104a1a1a1a10204b2b2b2b20000000200041000000001043c00000000000002\
             0000000c000000190000000200040000000001041d0000000300040100000001040900000002040800000002\
             0000000c000000190000000200040000000003040c0000000200040000000003040c00000000000100000008\
             00000002000400000000010446000000",
        ),
        (
            "depth",
            "050804200000000904040100000a04040100000104a1a1a1a10204b2b2b2b20001000000080000000200040000\
             0000010446000000",
        ),
        (
            "msaa4",
            "0404040400000008041c0000000104a1a1a1a10204b2b2b2b2000000010000000800000002000400000000010446\
             000000",
        ),
        ("nocolor", "010104a1a1a1a1"),
        ("nocolor4", "020404040000000104a1a1a1a1"),
        (
            "two",
            "030804140000000104a1a1a1a10204b2b2b2b200020000000c00000019000000020004000000000104460000000200\
             0401000000010446000000",
        ),
        (
            "amp2",
            "0408041c0000001a04020000000104a1a1a1a10204b2b2b2b2000000010000000800000002000400000000010446\
             000000",
        ),
        (
            "tess",
            "0408041c0000000c04010000000104a1a1a1a10204b2b2b2b2000000010000000800000002000400000000010446\
             000000",
        ),
        (
            "tess_idx",
            "050804200000000c04010000001004010000000104a1a1a1a10204b2b2b2b20001000000080000000200040000\
             0000010446000000",
        ),
        (
            "tile",
            "030404010000000304140000000104c3c3c3c300010000000800000002000400000000010446000000",
        ),
    ];

    /// `render_lib.py`'s `pipeline_blob(name, v, f, t)`: the blob, its sentinels replaced by refs.
    pub(crate) fn pipeline_blob(name: &str, v: u32, f: u32, t: u32) -> Vec<u8> {
        let (_, h) = BLOBS.iter().find(|(n, _)| *n == name).unwrap();
        let mut b = hex(h);
        for (sentinel, r) in [(0xa1u8, v), (0xb2, f), (0xc3, t)] {
            if let Some(at) = b.windows(4).position(|w| w == [sentinel; 4]) {
                b[at..at + 4].copy_from_slice(&r.to_le_bytes());
            }
        }
        b
    }

    fn decode_render(
        blob: &[u8],
        resolve: &mut Resolve,
    ) -> RaiseResult<Option<Decoded<RenderPipelineDescriptor>>> {
        render(Hold::NONE, blob, &mut |h, r| resolve.call(h, r))
    }

    fn decode_tile(
        blob: &[u8],
        resolve: &mut Resolve,
    ) -> RaiseResult<Option<Decoded<TileRenderPipelineDescriptor>>> {
        tile(Hold::NONE, blob, &mut |h, r| resolve.call(h, r))
    }

    /// The decoded render descriptor of `blob`, every function `f`, and the refs resolved.
    fn rendered(blob: &[u8], f: &VulkanFunction) -> (Decoded<RenderPipelineDescriptor>, Vec<u64>) {
        use reims_vgpu_core::platform::Handle as _;
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        let d = decode_render(blob, &mut resolve).unwrap().unwrap();
        (d, resolve.refs)
    }

    /// `render_blobs.m`'s `NEW(v, f)`: both functions, colour 0 `RGBA8Unorm`.
    fn new_render(f: &VulkanFunction) -> RenderPipelineDescriptor {
        let source = StageSource::Function(f.function_descriptor());
        let mut d = RenderPipelineDescriptor::new(source.clone());
        d.fragment = Some(source);
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d
    }

    /// Each of `render_lib.py`'s render blobs decodes to what `render_blobs.m` set, its functions
    /// resolved in the blob's order, and nothing ignored
    #[test]
    fn the_suites_render_blobs_decode_as_render_blobs_m_sets_them() {
        use metal2vulkan::metal::{
            BlendFactor, TessellationControlPointIndexType, TessellationPartitionMode,
        };
        let f = fixture_function(KERNEL);
        let source = StageSource::Function(f.function_descriptor());
        let blend = {
            let mut d = new_render(&f);
            let c = d.color_attachments[0].as_mut().unwrap();
            c.blending_enabled = true;
            c.source_rgb_blend_factor = BlendFactor::BlendColor;
            c.destination_rgb_blend_factor = BlendFactor::One;
            c.source_alpha_blend_factor = BlendFactor::BlendAlpha;
            c.destination_alpha_blend_factor = BlendFactor::Zero;
            d
        };
        let vdesc = {
            let mut d = new_render(&f);
            let mut v = VertexDescriptor::new();
            v.attributes.insert(
                0,
                VertexAttributeDescriptor::new(VertexFormat::Float2, 0, 0),
            );
            v.attributes.insert(
                1,
                VertexAttributeDescriptor::new(VertexFormat::UChar4Normalized, 8, 0),
            );
            v.layouts
                .insert(0, VertexBufferLayoutDescriptor::new(Some(12)));
            d.vertex_descriptor = Some(v);
            d
        };
        let depth = {
            let mut d = new_render(&f);
            d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
            d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
            d
        };
        let msaa4 = {
            let mut d = new_render(&f);
            d.raster_sample_count = 4;
            d
        };
        let nocolor = RenderPipelineDescriptor::new(source.clone());
        let nocolor4 = {
            let mut d = RenderPipelineDescriptor::new(source.clone());
            d.raster_sample_count = 4;
            d
        };
        let two = {
            let mut d = new_render(&f);
            d.color_attachments[1] = d.color_attachments[0];
            d
        };
        let amp2 = {
            let mut d = new_render(&f);
            d.max_vertex_amplification_count = 2;
            d
        };
        let mut tessellation = TessellationDescriptor::default();
        tessellation.partition_mode = TessellationPartitionMode::Integer;
        let tess = {
            let mut d = new_render(&f);
            d.tessellation = Some(tessellation);
            d
        };
        let tess_idx = {
            let mut d = tess.clone();
            let mut t = tessellation;
            t.control_point_index_type = TessellationControlPointIndexType::UInt16;
            d.tessellation = Some(t);
            d
        };
        for (name, want, refs) in [
            ("basic", new_render(&f), vec![0x21, 0x22]),
            ("blend", blend, vec![0x21, 0x22]),
            ("vdesc", vdesc, vec![0x21, 0x22]),
            ("depth", depth, vec![0x21, 0x22]),
            ("msaa4", msaa4, vec![0x21, 0x22]),
            ("nocolor", nocolor, vec![0x21]),
            ("nocolor4", nocolor4, vec![0x21]),
            ("two", two, vec![0x21, 0x22]),
            ("amp2", amp2, vec![0x21, 0x22]),
            ("tess", tess, vec![0x21, 0x22]),
            ("tess_idx", tess_idx, vec![0x21, 0x22]),
        ] {
            let (d, resolved) = rendered(&pipeline_blob(name, 0x21, 0x22, 0), &f);
            assert_eq!(d.desc, want, "{name}");
            assert_eq!(resolved, refs, "{name}");
            assert!(d.ignored.is_empty(), "{name}: {:?}", d.ignored);
        }
        // `vdesc_dyn`: the layout's stride dynamic
        let mut blob = pipeline_blob("vdesc", 0x21, 0x22, 0);
        while let Some(at) = blob.windows(6).position(|w| w == [3, 4, 0x0c, 0, 0, 0]) {
            blob[at + 2..at + 6].copy_from_slice(&DYNAMIC_STRIDE.to_le_bytes());
        }
        let (d, _) = rendered(&blob, &f);
        let v = d.desc.vertex_descriptor.unwrap();
        assert_eq!(v.layouts[&0].stride, None);
        assert_eq!(v.layouts.len(), 1);
    }

    /// The tile blob: its function, threadgroupSizeMatchesTileSize, colour 0 `RGBA8Unorm`
    #[test]
    fn the_suites_tile_blob_decodes_as_render_blobs_m_sets_it() {
        use reims_vgpu_core::platform::Handle as _;
        let f = fixture_function(KERNEL);
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        let d = decode_tile(&pipeline_blob("tile", 0, 0, 0x31), &mut resolve)
            .unwrap()
            .unwrap();
        let mut want =
            TileRenderPipelineDescriptor::new(StageSource::Function(f.function_descriptor()));
        want.color_attachment_formats[0] = Some(PixelFormat::RGBA8Unorm);
        want.threadgroup_size_matches_tile_size = true;
        assert_eq!(d.desc, want);
        assert_eq!(resolve.refs, vec![0x31]);
        assert!(d.ignored.is_empty());
        // Samples, tile buffers, the maximum threads, a label, and SPI key 8 at its default
        let blob = hex(
            "06 00 04 25000000 01 04 31000000 02 04 04000000 05 04 28000000 06 04 40000000 \
             08 04 00000000 6100 00 01000000 08000000 02 00 04 02000000 01 04 01000000",
        );
        let d = decode_tile(&blob, &mut resolve).unwrap().unwrap();
        assert_eq!(d.label.as_deref(), Some("a"));
        let d = d.desc;
        assert_eq!(d.raster_sample_count, 4);
        assert_eq!(d.max_total_threads_per_threadgroup, Some(64));
        assert_eq!(d.buffers, BTreeMap::from([(2, Mutability::Mutable)]));
        assert_eq!(d.color_attachment_formats, [None; 8]);
    }

    /// Render's other keys: alpha to coverage and to one, the topology, indirect command buffer
    /// support, buffer mutabilities, every colour attachment field, a fragment ref that resolves
    /// to nil (none, the decode going on), rasterization off (no fragment), a label, an SPI key at
    /// its default, and undefined keys at the top and in an attachment
    #[test]
    fn render_keys_decode_as_metal_holds_them() {
        use metal2vulkan::metal::{BlendFactor, BlendOperation, ColorWriteMask};
        let f = fixture_function(KERNEL);
        // 5, 6, 11 (Triangle), 25, 21 (vertexBuffers[1] Immutable), then the functions
        let blob = hex(
            "07 05 04 01000000 06 04 03000000 0b 04 03000000 19 04 01000000 15 04 2c000000 \
             01 04 21000000 02 04 22000000 00 01000000 08000000 02 00 04 01000000 01 04 02000000",
        );
        let (d, _) = rendered(&blob, &f);
        let d = d.desc;
        assert!(d.alpha_to_coverage && d.alpha_to_one && d.support_indirect_command_buffers);
        assert_eq!(d.input_primitive_topology, PrimitiveTopologyClass::Triangle);
        assert_eq!(
            d.vertex_buffers,
            BTreeMap::from([(1, Mutability::Immutable)])
        );
        assert!(d.fragment_buffers.is_empty());
        // Colour 7 of every field, then an undefined element key; colour 3 with no format
        let blob = hex("03 08 04 14000000 01 04 21000000 02 04 22000000 00 \
             02000000 0c000000 4f000000 \
             0b 00 04 07000000 01 04 46000000 02 04 01000000 03 04 04000000 04 04 05000000 \
             05 04 01000000 06 04 08000000 07 04 09000000 08 04 04000000 09 04 0c000000 \
             0a 04 01000000 \
             02 00 04 03000000 02 04 01000000");
        let (d, _) = rendered(&blob, &f);
        let mut c = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        c.blending_enabled = true;
        c.source_rgb_blend_factor = BlendFactor::SourceAlpha;
        c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        c.rgb_blend_operation = BlendOperation::Subtract;
        c.source_alpha_blend_factor = BlendFactor::DestinationAlpha;
        c.destination_alpha_blend_factor = BlendFactor::OneMinusDestinationAlpha;
        c.alpha_blend_operation = BlendOperation::Max;
        c.write_mask = ColorWriteMask::RED | ColorWriteMask::GREEN;
        let mut want = [None; 8];
        want[7] = Some(c);
        assert_eq!(d.desc.color_attachments, want);
        assert_eq!(
            format!("{:?}", d.ignored),
            "[render.colorAttachments.field_0a = [01, 00, 00, 00]]"
        );
        // A fragment ref that resolves to nil is none; the decode goes on
        use reims_vgpu_core::platform::Handle as _;
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        resolve.nil.push(0x99);
        let d = decode_render(
            &hex("03 01 04 21000000 02 04 99000000 04 04 02000000"),
            &mut resolve,
        )
        .unwrap()
        .unwrap()
        .desc;
        assert_eq!((d.fragment, d.raster_sample_count), (None, 2));
        assert_eq!(resolve.refs, vec![0x21, 0x99]);
        // Rasterization off: the fragment function is resolved and kept, for the runtime to check
        let (d, refs) = rendered(&hex("03 07 04 00000000 01 04 21000000 02 04 22000000"), &f);
        assert!(!d.desc.rasterization_enabled);
        assert_eq!((d.desc.fragment.is_some(), refs), (true, vec![0x21, 0x22]));
        // A label, SPI keys at their default 0, and undefined keys 19 and 0x7f
        let (d, _) = rendered(
            &hex(
                "06 00 04 25000000 2d 04 00000000 18 04 00000000 13 04 05000000 7f 04 06000000 01 04 21000000 616200",
            ),
            &f,
        );
        let keys: Vec<_> = d.ignored.iter().map(|i| (i.block, i.key)).collect();
        assert_eq!(keys, vec![("render", 0x13), ("render", 0x7f)]);
        assert_eq!(d.label.as_deref(), Some("ab"));
        // Tessellation at Metal's defaults is none; each of its keys sets its field
        let (d, _) = rendered(&hex("02 0d 04 10000000 01 04 21000000"), &f);
        assert_eq!(d.desc.tessellation, None);
        let (d, _) = rendered(
            &hex(
                "07 0c 04 02000000 0d 04 08000000 0e 04 01000000 10 04 02000000 11 04 01000000 \
                 12 04 01000000 01 04 21000000",
            ),
            &f,
        );
        let t = d.desc.tessellation.unwrap();
        assert_eq!(t.max_tessellation_factor, 8);
        assert!(t.factor_scale_enabled);
        assert_eq!(
            format!(
                "{:?}",
                (
                    t.partition_mode,
                    t.control_point_index_type,
                    t.factor_step_function,
                    t.output_winding_order
                )
            ),
            "(FractionalOdd, UInt32, PerPatch, CounterClockwise)"
        );
    }

    /// The suite's garbage blob, `render_objects_malformed_garbage`'s.
    fn garbage() -> Vec<u8> {
        (0..64u32).map(|i| ((i * 37 + 11) & 0xff) as u8).collect()
    }

    /// The render form dies where Metal aborts, at its selector: the suite's garbage, truncated
    /// and nil-vertex blobs (the fragment resolved first), a colour index past 7, no vertex
    /// function, an SPI key at a value the runtime cannot give; an empty blob is nil
    #[test]
    fn render_blobs_metal_aborts_on_die_at_the_selector() {
        use reims_vgpu_core::platform::Handle as _;
        let f = fixture_function(KERNEL);
        let basic = pipeline_blob("basic", 0x21, 0x22, 0);
        let color = |index: u32| {
            let mut b = hex("02 08 04 10000000 01 04 21000000 000000 01000000 08000000 02 00 04");
            b.extend(index.to_le_bytes());
            b.extend(hex("01 04 46000000"));
            b
        };
        for (what, blob) in [
            ("render_objects_malformed_garbage", garbage()),
            ("render_objects_malformed_truncated", basic[..5].to_vec()),
            ("colour index 8", color(8)),
            ("no vertex function", hex("01 02 04 22000000")),
            ("sampleMask 0", hex("02 1c 04 00000000 01 04 21000000")),
            ("alphaTestEnabled", hex("02 2d 04 01000000 01 04 21000000")),
            (
                "a value of 8 bytes",
                hex("02 04 08 0400000000000000 01 04 21000000"),
            ),
        ] {
            let mut resolve = Resolve::answering(Some(f.clone_ref()));
            let site = catch_death(|| decode_render(&blob, &mut resolve).is_ok()).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, RENDER_SEL),
                "{what}"
            );
        }
        rendered(&color(7), &f);
        // render_objects_malformed_nil_vertex: the fragment is resolved, then the decode dies
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        resolve.nil.push(0);
        let blob = pipeline_blob("basic", 0, 0x22, 0);
        let site = catch_death(|| decode_render(&blob, &mut resolve).is_ok()).unwrap_err();
        assert_eq!(site.file, RENDER_SEL);
        assert_eq!(resolve.refs, vec![0, 0x22]);
        let mut resolve = Resolve::answering(None);
        assert!(decode_render(&[], &mut resolve).unwrap().is_none());
        assert!(resolve.refs.is_empty());
    }

    /// A render blob's colour, depth or stencil attachment, or a tile blob's colour attachment,
    /// of a pixel format the runtime holds raw decodes, the first of them kept for the state's
    /// creation ([`RawFormat`]), and a later entry's function is resolved, its raise the guest's,
    /// as Metal's deserializer takes the format; a tile blob's colour attachment of a value
    /// Metal's setter does not take dies at the deserializer, where Metal asserts
    #[test]
    fn raw_pixel_formats_decode_for_the_states_creation() {
        use reims_vgpu_core::platform::Handle as _;
        let f = fixture_function(KERNEL);
        let colour = |head: &str, format: u32| {
            let mut b = hex(head);
            b.extend(hex("01000000 08000000 02 00 04 00000000 01 04"));
            b.extend(format.to_le_bytes());
            b
        };
        let raw = |attachment: &str, raw| {
            Some(RawFormat {
                attachment: attachment.to_owned(),
                raw,
            })
        };
        for format in [2000, 630, 255, 262] {
            let blob = colour("02 08 04 10000000 01 04 21000000 000000", format);
            let (d, _) = rendered(&blob, &f);
            let attachment = "colour attachment 0";
            assert_eq!(d.raw_format, raw(attachment, u64::from(format)));
            assert!(d.desc.color_attachments[0].is_none());
            for (key, attachment) in [("09", "depth attachment"), ("0a", "stencil attachment")] {
                let mut blob = hex(&format!("02 01 04 21000000 {key} 04"));
                blob.extend(format.to_le_bytes());
                let (d, _) = rendered(&blob, &f);
                assert_eq!(d.raw_format, raw(attachment, u64::from(format)));
            }
        }
        // The colour attachment comes first; a later entry's function raises.
        let blob = colour("02 08 04 10000000 01 04 21000000 000000", 630);
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        resolve.raises.push(0x21);
        assert!(decode_render(&blob, &mut resolve).is_err());
        for format in [630, 255, 262] {
            let blob = colour("02 03 04 10000000 01 04 31000000 000000", format);
            let mut resolve = Resolve::answering(Some(f.clone_ref()));
            let d = decode_tile(&blob, &mut resolve).unwrap().unwrap();
            assert_eq!(d.raw_format, raw("colour attachment 0", u64::from(format)));
        }
        let blob = colour("02 03 04 10000000 01 04 31000000 000000", 2000);
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        let site = catch_death(|| decode_tile(&blob, &mut resolve).is_ok()).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, TILE_SEL));
    }

    /// The tile form dies where Metal aborts, at its selector: garbage, a truncated blob, a
    /// colour index past 7, no tile function, an SPI key at a value the runtime cannot give; an
    /// empty blob is nil
    #[test]
    fn tile_blobs_metal_aborts_on_die_at_the_selector() {
        use reims_vgpu_core::platform::Handle as _;
        let f = fixture_function(KERNEL);
        let color = |index: u32| {
            let mut b = hex("02 03 04 10000000 01 04 31000000 000000 01000000 08000000 02 00 04");
            b.extend(index.to_le_bytes());
            b.extend(hex("01 04 46000000"));
            b
        };
        for (what, blob) in [
            ("garbage", garbage()),
            ("colour index 8", color(8)),
            ("no tile function", hex("01 04 04 01000000")),
            (
                "textureWriteRoundingMode 3",
                hex("02 08 04 03000000 01 04 31000000"),
            ),
            ("truncated", pipeline_blob("tile", 0, 0, 0x31)[..9].to_vec()),
        ] {
            let mut resolve = Resolve::answering(Some(f.clone_ref()));
            let site = catch_death(|| decode_tile(&blob, &mut resolve).is_ok()).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, TILE_SEL),
                "{what}"
            );
        }
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        assert!(decode_tile(&color(7), &mut resolve).unwrap().is_some());
        let mut resolve = Resolve::answering(None);
        assert!(decode_tile(&[], &mut resolve).unwrap().is_none());
        assert!(resolve.refs.is_empty());
    }

    /// Metal's SPI keys the runtime has no field for die naming the property; a value of a size
    /// other than 4 dies naming its key
    #[test]
    fn keys_with_no_home_die_naming_the_property() {
        let f = fixture_function(KERNEL);
        use reims_vgpu_core::platform::Handle as _;
        for h in [
            "02 0a 04 03000000 00 04 11000000",
            "02 0c 04 01000000 00 04 11000000",
            "02 00 04 11000000 08 05 4000000000",
        ] {
            let site = dies(&hex(h), &mut Resolve::answering(Some(f.clone_ref())));
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, COMPUTE_SEL),
                "{h}"
            );
        }
        // At Metal's default, 0, they decode to nothing
        for h in [
            "02 0a 04 00000000 00 04 11000000",
            "02 0c 04 00000000 00 04 11000000",
        ] {
            let (d, _) = decoded(&hex(h), &f);
            assert!(d.ignored.is_empty(), "{h}");
        }
        // A raw value the runtime's type cannot hold: a mutability of 7
        let blob = hex(
            "02 06 04 10000000 00 04 11000000 000000 01000000 08000000 02 00 04 01000000 01 04 07000000",
        );
        let site = dies(&blob, &mut Resolve::answering(Some(f.clone_ref())));
        assert_eq!(site.file, COMPUTE_SEL);
    }

    /// The render form's SPI keys are decided once the block is read, each key's last value
    /// winning, in any order: the values Metal draws as without them decode to nothing, and the
    /// rest die at the selector, after every function entry has been resolved, so a resolver's
    /// raise after them is the caller's
    #[test]
    fn spi_keys_are_decided_once_the_block_is_read() {
        use reims_vgpu_core::platform::Handle as _;
        let f = fixture_function(KERNEL);
        let blends: &[(u8, u32)] = &[(2, 1), (3, 4), (4, 5)];
        let blob = |entries: &[&str], colour: &[(u8, u32)]| {
            let mut element = vec![(0, 0), (1, 70)];
            element.extend_from_slice(colour);
            rounded_render_of(entries, &[&element], 0)
        };
        let defaults = [
            "18 04 00000000",
            "1b 04 0000803f",
            "2d 04 00000000",
            "2e 04 00000000",
            "2f 04 00000000",
            "31 04 00000000",
            "32 04 00000000",
            "33 04 00000000",
            "34 04 00000000",
            "35 04 00000000",
            "36 04 00000000",
            "37 04 00000000",
            "38 04 00000000",
            "39 04 00000000",
            "3a 04 ffffffff",
        ];
        for entries in [
            &defaults[..],
            &["1c 04 ffffffff"],
            &["1c 04 01000000"],
            &["1c 04 0f000000", "04 04 04000000"],
            &["04 04 04000000", "1c 04 1f000000"],
            &["1c 04 00000000", "1c 04 03000000"],
            &["30 04 00000000"],
            &["30 04 ffffffff"],
            &["38 04 01000000"],
            &["38 04 ffffffff"],
            &["2e 04 04000000"],
            &["2d 04 01000000", "2e 04 07000000"],
            &["2e 04 07000000", "2d 04 02000000"],
            &[
                "2d 04 01000000",
                "2e 04 07000000",
                "2d 04 00000000",
                "2e 04 00000000",
            ],
            &["37 04 0a000000"],
            &["2f 04 01000000", "37 04 02000000"],
            &["37 04 02000000", "2f 04 01000000"],
        ] {
            let (d, _) = rendered(&blob(entries, &[]), &f);
            assert!(d.ignored.is_empty(), "{entries:?}");
        }
        let (d, _) = rendered(&blob(&["37 04 0a000000"], blends), &f);
        assert!(d.desc.color_attachments[0].unwrap().blending_enabled);
        type Dying<'a> = (Vec<&'a str>, &'a [(u8, u32)]);
        let mut dying: Vec<Dying<'_>> = vec![
            (vec!["1c 04 00000000"], &[]),
            (vec!["1c 04 02000000"], &[]),
            (vec!["1c 04 07000000", "04 04 04000000"], &[]),
            (vec!["04 04 21000000", "1c 04 ffffffff"], &[]),
            (vec!["1c 04 ffffffff", "1c 04 00000000"], &[]),
            (vec!["1b 04 0000003f"], &[]),
            (vec!["1b 04 00000040"], &[]),
            (vec!["3a 04 00000000"], &[]),
            (vec!["2d 04 01000000"], &[]),
            (vec!["2d 04 01000000", "2e 04 04000000"], &[]),
            (
                vec!["2e 04 07000000", "2d 04 01000000", "2e 04 00000000"],
                &[],
            ),
            (vec!["2f 04 01000000"], &[]),
            (vec!["2f 04 01000000", "37 04 0a000000"], &[]),
            (vec!["2f 04 01000000", "37 04 02000000"], blends),
        ];
        for key in [0x18, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x39] {
            let entry: &str = Box::leak(format!("{key:02x} 04 01000000").into_boxed_str());
            dying.push((vec![entry], &[]));
        }
        for (entries, colour) in dying {
            let mut resolve = Resolve::answering(Some(f.clone_ref()));
            let blob = blob(&entries, colour);
            let site = catch_death(|| decode_render(&blob, &mut resolve).is_ok()).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, RENDER_SEL),
                "{entries:?}"
            );
        }
        // A function entry after a key that dies is resolved before the death, and its raise is
        // the caller's
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        let blob = hex("03 2d 04 01000000 01 04 21000000 02 04 22000000");
        catch_death(|| decode_render(&blob, &mut resolve).is_ok()).unwrap_err();
        assert_eq!(resolve.refs, vec![0x21, 0x22]);
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        resolve.raises.push(0x22);
        let r = decode_render(&blob, &mut resolve).err().unwrap();
        assert_eq!(r.detail(), Some("lookup"));
    }

    /// The little-endian bytes of `v`, as hex.
    pub(crate) fn le(v: u32) -> String {
        v.to_le_bytes().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// A render blob of vertex ref 1, fragment ref 2, colour 0 of `format` and
    /// `textureWriteRoundingMode` `mode`.
    pub(crate) fn rounded_render(mode: u32, format: u32) -> Vec<u8> {
        rounded_render_of(&["02 04 02000000"], &[&[(0, 0), (1, format)]], mode)
    }

    /// A render blob of vertex ref 1, the 4-byte `entries`, `textureWriteRoundingMode` `mode`,
    /// and an element of the colour attachments for each of `colours`, of its 4-byte entries
    /// (key 0 its index, 1 its pixel format, 9 its write mask).
    pub(crate) fn rounded_render_of(
        entries: &[&str],
        colours: &[&[(u8, u32)]],
        mode: u32,
    ) -> Vec<u8> {
        let count = entries.len() + 3;
        let end = 1 + 6 * count;
        let at = end.next_multiple_of(4);
        let mut blob = hex(&format!(
            "{count:02x} 08 04 {} 01 04 01000000 {} 1e 04 {}",
            le(u32::try_from(at).unwrap()),
            entries.join(" "),
            le(mode),
        ));
        blob.resize(at, 0);
        let elements: Vec<Vec<u8>> = colours
            .iter()
            .map(|entries| {
                let mut b = vec![u8::try_from(entries.len()).unwrap()];
                for &(key, value) in *entries {
                    b.extend([key, 4]);
                    b.extend(value.to_le_bytes());
                }
                b
            })
            .collect();
        blob.extend(u32::try_from(colours.len()).unwrap().to_le_bytes());
        let mut offset = 4 + 4 * colours.len();
        for element in &elements {
            blob.extend(u32::try_from(offset).unwrap().to_le_bytes());
            offset += element.len();
        }
        blob.extend(elements.concat());
        blob
    }

    /// A tile blob of tile function ref 3, colour 0 of `format` and `textureWriteRoundingMode`
    /// `mode`.
    fn rounded_tile(mode: u32, format: u32) -> Vec<u8> {
        hex(&format!(
            "04 04 04 01000000 03 04 1c000000 01 04 03000000 08 04 {} 000000 \
             01000000 08000000 02 00 04 00000000 01 04 {}",
            le(mode),
            le(format)
        ))
    }

    /// A draw under a mode 2 state whose fragment function may write textures names what may
    /// write a texture of one of [`MODE_2_TEXTURE_FORMATS`]: one bound at a slot the function
    /// writes, and any where the function takes handles, writes no slot, or its reflection cannot
    /// tell; one bound at a slot the function does not write runs, and so does every draw with
    /// none bound
    #[test]
    fn rounded_texture_writes_name_what_may_write_a_half_float_texture() {
        use crate::gpu::writes::Slots;

        let mut writes = Slots::default();
        writes.textures.push(0..2);
        let at = |slot| BTreeMap::from([(slot, "RGBA16Float")]);
        let none = BTreeMap::new();
        let named = rounded_texture_writes(Ok(&writes), &at(1)).unwrap();
        assert!(named.starts_with("texture 1 of RGBA16Float"), "{named}");
        assert_eq!(rounded_texture_writes(Ok(&writes), &at(2)), None);
        assert_eq!(rounded_texture_writes(Ok(&writes), &none), None);
        let handles = Slots {
            handles: true,
            ..writes.clone()
        };
        for untold in [
            rounded_texture_writes(Ok(&handles), &none),
            rounded_texture_writes(Ok(&Slots::default()), &none),
            rounded_texture_writes(Err("a refused reflection"), &none),
        ] {
            assert!(untold.is_some());
        }
    }

    /// A mode 2 draw decides its fragment function's texture writes of the four half-float
    /// formats, `RG11B10Float` and `RGB9E5Float`, whose writes the runtime takes no mode for; a
    /// mode 2 state decides its colour attachments of the three half-float formats the runtime
    /// rounds to nearest even, each also a texture format a draw decides
    #[test]
    fn mode_2_decides_the_formats_it_may_write_otherwise() {
        use PixelFormat as F;
        let names = |formats: &[PixelFormat]| formats.iter().map(|f| f.name()).collect::<Vec<_>>();
        assert_eq!(
            names(&MODE_2_TEXTURE_FORMATS),
            names(&[
                F::R16Float,
                F::RG16Float,
                F::RGBA16Float,
                F::RGBX16Float,
                F::RG11B10Float,
                F::RGB9E5Float
            ])
        );
        assert_eq!(
            names(&MODE_2_ATTACHMENT_FORMATS),
            names(&[F::R16Float, F::RG16Float, F::RGBA16Float])
        );
        assert!(
            MODE_2_ATTACHMENT_FORMATS
                .iter()
                .all(|f| MODE_2_TEXTURE_FORMATS.contains(f))
        );
    }

    /// A render state of mode 2 whose fragment function writes a half-float colour attachment
    /// as written names that attachment where its sample count is one whose resolve at the mode
    /// nothing measured (8), as Metal's state then differs from the runtime's; of 1 sample, or of
    /// a count measured (2 and 4), it writes as the runtime's, and so does every count at mode 0,
    /// whose state the creator asks nothing of its rounding (`states`' render state creator asks
    /// only one decoded to nearest even)
    #[test]
    fn a_mode_2_state_of_an_unmeasured_sample_count_names_its_resolve() {
        let f = fixture_function(KERNEL);
        let colour: &[(u8, u32)] = &[(0, 0), (1, 115)];
        let of = |samples: u32, mode| {
            let count = format!("04 04 {}", le(samples));
            let (d, _) = rendered(
                &rounded_render_of(&["02 04 02000000", &count], &[colour], mode),
                &f,
            );
            assert_eq!(d.desc.raster_sample_count, samples);
            assert_eq!(d.nearest_even, mode == NEAREST_EVEN, "{mode}");
            // As the creator decides it: only a state decoded to nearest even.
            d.nearest_even
                .then(|| rounded_writes(&d.desc, &[0], false))
                .flatten()
        };
        let named = of(8, NEAREST_EVEN).unwrap_or_default();
        assert!(
            named.starts_with("colour attachment 0 of MTLPixelFormatRGBA16Float, of 8 samples"),
            "{named}"
        );
        for samples in [1, 2, 4] {
            assert_eq!(of(samples, NEAREST_EVEN), None, "{samples}");
        }
        for samples in [1, 2, 4, 8] {
            assert_eq!(of(samples, 0), None, "{samples}");
        }
    }

    /// `textureWriteRoundingMode` decodes to nothing from 0 to 2 in each form, half-float colour
    /// attachments included, but a render blob's: its 2 is the runtime's nearest even, and is
    /// kept for its state, decided once made, its 0 and 1 toward zero; a mode past 2 dies in
    /// each form
    #[test]
    fn texture_write_rounding_modes_decode_to_nothing_where_they_change_nothing() {
        use reims_vgpu_core::platform::Handle as _;
        let f = fixture_function(KERNEL);
        let formats = [10, 25, 65, 70, 115];
        for mode in ROUNDING_MODES {
            let (d, _) = decoded(&hex(&format!("02 0a 04 {} 00 04 11000000", le(mode))), &f);
            assert!(d.ignored.is_empty() && !d.nearest_even, "{mode}");
            for format in formats {
                let mut resolve = Resolve::answering(Some(f.clone_ref()));
                let d = decode_tile(&rounded_tile(mode, format), &mut resolve)
                    .unwrap()
                    .unwrap();
                assert!(d.ignored.is_empty() && !d.nearest_even, "{mode} {format}");
                // No fragment function, or rasterization off, under which none rasterizes
                for (entries, fragment) in [
                    (&["07 04 01000000"][..], false),
                    (&["02 04 02000000", "07 04 00000000"], true),
                ] {
                    let colour: &[(u8, u32)] = &[(0, 0), (1, format)];
                    let (d, _) = rendered(&rounded_render_of(entries, &[colour], mode), &f);
                    let kept = d.desc.fragment.is_some();
                    assert_eq!(kept, fragment, "{mode} {format} {entries:?}");
                    assert_eq!(
                        rounded_writes(&d.desc, &[0], false),
                        None,
                        "{mode} {format}"
                    );
                    assert_eq!(d.nearest_even, mode == NEAREST_EVEN, "{mode} {format}");
                }
                let (d, _) = rendered(&rounded_render(mode, format), &f);
                assert!(d.ignored.is_empty(), "{mode} {format}");
                assert!(d.desc.fragment.is_some(), "{mode} {format}");
                assert_eq!(d.nearest_even, mode == NEAREST_EVEN, "{mode} {format}");
                let rounding = match mode {
                    NEAREST_EVEN => TextureWriteRoundingMode::NearestEven,
                    _ => TextureWriteRoundingMode::TowardZero,
                };
                assert_eq!(d.desc.texture_write_rounding_mode, rounding, "{mode}");
            }
        }
        let render_dies = |blob: &[u8]| {
            let mut resolve = Resolve::answering(Some(f.clone_ref()));
            catch_death(|| decode_render(blob, &mut resolve).is_ok()).unwrap_err()
        };
        let tile_dies = |blob: &[u8]| {
            let mut resolve = Resolve::answering(Some(f.clone_ref()));
            catch_death(|| decode_tile(blob, &mut resolve).is_ok()).unwrap_err()
        };
        for mode in [3, u32::MAX] {
            let site = dies(
                &hex(&format!("02 0a 04 {} 00 04 11000000", le(mode))),
                &mut Resolve::answering(Some(f.clone_ref())),
            );
            assert_eq!(site.file, COMPUTE_SEL, "{mode}");
            assert_eq!(render_dies(&rounded_render(mode, 70)).file, RENDER_SEL);
            assert_eq!(tile_dies(&rounded_tile(mode, 70)).file, TILE_SEL);
        }
    }
}
