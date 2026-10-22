//! `MTLPixelFormatGetInfoForDevice()` and `MTLGetTextureLevelInfoForDeviceWithOptions()`, the
//! runtime's answers.
//!
//! **A pixel format's description** ([`description`]), which core gives the guest (its flags, of
//! which it reads the compressed, PVRTC and two-plane bits, and its four block words,
//! `reims_vgpu_core::resources::texture::pixel_format`), is the runtime's
//! (`Device::pixel_format_info`): Apple's private description at the device's Metal level, from
//! the runtime's own measurements of plain Metal at every value from 0 to 2047 on an Apple M3 Max
//! under macOS 26.7 and under 27.0.1 (26A434), each of which describes the same 264 formats (an
//! Apple M4 under 27.0.1 describes 630 to 645 too, which neither level does). A value the level
//! describes no format for is described as
//! Metal describes it: `MTLPixelFormatInvalid`, every word 0. No level describes
//! `Depth24Unorm_Stencil8` (255) or `X24_Stencil8` (262), as no Apple GPU has them. A format's
//! description does not hang on whether the runtime makes textures of it: a descriptor takes one
//! it lacks, and a texture of it dies where it is made ([`crate::gpu::desc::lacked`]) or is nil
//! ([`crate::gpu::no_texture`]). At macOS 26.7 the descriptions are the Metal platform's on this
//! host at every value (`descriptions_are_the_metal_platforms_here`), and so are they at 27.0.1 on
//! this host under 27.0.1. The two levels describe a value otherwise, in what core reads, only in
//! the flags of five formats
//! (`RGB10A2Unorm_sRGB`, `_PQ` and `_HLG`, 586, 587 and 623, and `BGR10A2Unorm_sRGB` and `_PQ`, 628
//! and 629); macOS 27.0.1 also numbers 175 of the 264 formats' cast classes anew, which core does
//! not read (`descriptions_differ_by_level_only_where_the_runtime_measured_it`).
//!
//! **A texture level's layout** ([`level_layout`]), which core checks a texture on an IOSurface
//! against (its rows' bytes, its rows, its bytes a pixel), is the runtime's
//! (`Device::texture_level_info`): computed from the description by the rule the runtime fitted,
//! with no miss, to 34,306 of Metal's answers on the M3 Max and 34,626 on the M4, over
//! every described format at 20 sizes, levels (the sixth argument) from 0 to 2^32, depths, array
//! lengths, plane arguments (the seventh), values no format is, and zero sizes, which Metal
//! answers as sizes of 1. `levels_are_the_metal_platforms_here` compares it with the Metal
//! platform on this host, where it wraps at 2^64 as Metal does: it counts an extent's blocks as
//! `(extent + block - 1) / block` with an add that wraps, so a width of `u64::MAX` has no blocks
//! 4 wide (`wrapped_block_counts_are_metals`), and its products of blocks, bytes, depths and
//! array lengths wrap (at extents of 2^62 + 3 and `u64::MAX`, and depths and array lengths of
//! 2^32 + 5 and 2^62 + 3). Core asks none of these: its extents are the guest's 32-bit width and
//! height, at depth 1 and array length 1.
//!
//! **What this platform's own rules read of a description** ([`block`]: a fill's texel, a
//! texture's bytes by Apple's layout, a CPU write's unit) is macOS 26.7's, where those rules were
//! measured; the levels describe it alike.
//!
//! **Which values Metal's setter takes and which formats Metal describes** differ by macOS
//! release, so both are the runtime's row of the release the devices answer as
//! ([`crate::platform::metal_level`]: [`setter_takes`], [`described`]), which at macOS 26.7 is
//! the measured host's (`the_measured_host_is_the_runtimes_level_of_macos_26_7`). On a host whose
//! release the runtime has not measured, whose devices answer as the newest measured release below
//! it, an answer the measured releases give otherwise dies where it is asked
//! ([`crate::platform::answering_level`]): the setter's at the values only one release takes,
//! whether Metal describes a format where a view's nil or a state's attachment hangs on it
//! ([`describes_answer`], [`crate::gpu::held_format`]), whether a creator makes a texture of it
//! ([`crate::gpu::desc::TextureFields::descriptor`]), a description whose flags or words differ
//! ([`description`]; the cast class, which core does not read, is the level's) and a level's
//! layout that differs ([`level_layout`]). The setter holds such a format as the level describes
//! it, which the guest does not see ([`crate::gpu::set_format`]).

use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{self as rt, MetalLevel};
use reims_vgpu_core::platform::types::{PixelFormatInfo, TextureLevelInfo, TextureLevelQuery};

/// Whether Metal describes `raw` as a format: a format the runtime has that the devices' Metal
/// level describes ([`MetalLevel::describes`], [`crate::platform::metal_level`]), which no level
/// does of `Depth24Unorm_Stencil8` (255) and `X24_Stencil8` (262), as no Apple GPU has them. At
/// macOS 26.7 these are the 264 formats the Metal platform describes on this
/// host (`the_measured_host_is_the_runtimes_level_of_macos_26_7`).
pub(crate) fn described(sel: &'static str, raw: u64) -> bool {
    described_at(|| crate::platform::metal_level(sel), raw)
}

/// [`described`], where the answer `sel` gives depends on it, as it does where a view is nil or a
/// state's attachment hangs on it: at a level below the host's release, where the levels the
/// runtime measured differ for `raw`, `sel` dies ([`crate::platform::answering_level`]).
/// [`described`] itself picks how a value is held, which the guest does not see, and never dies.
pub(crate) fn describes_answer(sel: &'static str, raw: u64) -> bool {
    let what = || format!("whether Metal describes pixel format {raw}");
    described_at(|| crate::platform::answering_level(sel, what), raw)
}

/// Whether the Metal levels the runtime measured describe `raw` otherwise, one a format and
/// another none ([`MetalLevel::describes`]), where [`described`] asks the level: no value at the
/// runtime's levels, which describe the same formats.
pub(crate) fn level_decides(raw: u64) -> bool {
    crate::gpu::runtime_pixel_format(raw).is_some_and(|pf| {
        let mut answers = MetalLevel::ALL.into_iter().map(|l| l.describes(pf));
        let first = answers.next();
        answers.any(|a| Some(a) != first)
    })
}

/// [`described`] at the level `level` answers, which is asked only where the levels the runtime
/// measured differ for `raw`.
fn described_at(level: impl FnOnce() -> MetalLevel, raw: u64) -> bool {
    at_level(level, raw, MetalLevel::describes)
}

/// Whether Metal's `setPixelFormat:` takes `raw`: a value the devices' Metal level knows
/// ([`MetalLevel::knows`], [`crate::platform::metal_level`]), `MTLPixelFormatInvalid` among them,
/// of a format the runtime has. Measured on the M3 Max (macOS 26.7), one value a process, at
/// every value from 0 to 2047, without `MTL_DEBUG_LAYER`: the setter takes the formats Metal
/// describes, `MTLPixelFormatInvalid`, 255, 262 and 630 to 650, and asserts at every other value
/// ("pixelFormat (2000) is not a valid MTLPixelFormat"), the 13 three-channel formats among them;
/// every texture creator asserts at a descriptor of one of those it takes but describes none for
/// ("MTLTextureDescriptor has invalid pixelFormat (630)"), but
/// `newSharedTextureWithDescriptor:`, which answers nil for one at a storage other than
/// `Private`, and at `Private` for 0 and 630 to 650
/// (`formats_the_runtime_lacks_die_where_textures_are_made`). That is the runtime's row of macOS
/// 26.7 (`the_measured_host_is_the_runtimes_level_of_macos_26_7`); macOS 27.0.1's setter takes
/// the three-channel formats too, which it describes none of.
pub(crate) fn setter_takes(sel: &'static str, raw: u64) -> bool {
    let what = || format!("whether Metal's setPixelFormat: takes {raw}");
    at_level(
        || crate::platform::answering_level(sel, what),
        raw,
        |level, pf| level.knows(pf),
    )
}

/// `answer` for the runtime's format of `raw` at the level `level` gives, false for a value the
/// runtime has no format of. Where every level the runtime measured answers alike, that answer,
/// without asking `level`: the levels differ only at the three-channel formats, which macOS
/// 27.0.1's setter takes and 26.7's does not.
fn at_level(
    level: impl FnOnce() -> MetalLevel,
    raw: u64,
    answer: impl Fn(MetalLevel, PixelFormat) -> bool,
) -> bool {
    let Some(pf) = crate::gpu::runtime_pixel_format(raw) else {
        return false;
    };
    let mut answers = MetalLevel::ALL.into_iter().map(|l| answer(l, pf));
    let first = answers.next().unwrap_or(false);
    if answers.all(|a| a == first) {
        first
    } else {
        answer(level(), pf)
    }
}

/// `flags` bit 10: a format of compressed blocks (core's `KIND_COMPRESSED`), whose block words
/// are its block's width and height.
pub(crate) const COMPRESSED: u64 = 0x400;

/// `MTLPixelFormatGetInfoForDevice()` (`sel`) of `raw` on `device`: Metal's description at the
/// device's Metal level, the runtime's (`Device::pixel_format_info`), field for field. Where the
/// levels the runtime measured describe `raw` otherwise in its name, flags or words, the level is
/// asked, which on a host whose release the runtime has not measured dies at `sel`
/// ([`crate::platform::answering_level`]): five formats whose flags differ (the module's docs).
/// The cast class, which 27.0.1 numbers anew
/// for most formats, is the device's level's there too, as core does not read it.
pub(crate) fn description(sel: &'static str, device: &rt::Device, raw: u64) -> PixelFormatInfo {
    if !alike(raw, |info| (info.name, info.flags, info.words)) {
        crate::platform::answering_level(sel, || {
            format!("Metal's description of pixel format {raw}")
        });
    }
    let info = device.pixel_format_info(raw);
    let [block_bytes, block_width, block_height, block_depth] = info.words;
    PixelFormatInfo {
        name: Some(info.name.to_owned()),
        flags: info.flags,
        cast_class: info.cast_class,
        block_bytes,
        block_width,
        block_height,
        block_depth,
    }
}

/// `MTLGetTextureLevelInfoForDeviceWithOptions()` (`sel`) of `q` on `device`: Metal's layout of
/// the level at the device's Metal level, the runtime's (`Device::texture_level_info`), word for
/// word. Core's sixth and seventh arguments (`options` and `flags`) are the
/// runtime's level and plane argument, as its probes found Metal reads them. Where the levels the
/// runtime measured lay the query out otherwise (a format one describes and another does not,
/// which no value is at the runtime's levels), the level is asked, which on a host whose release
/// the runtime has not measured dies at `sel` ([`crate::platform::answering_level`]).
pub(crate) fn level_layout(
    sel: &'static str,
    device: &rt::Device,
    q: &TextureLevelQuery,
) -> TextureLevelInfo {
    let query = rt::TextureLevelQuery {
        pixel_format: q.pixel_format,
        width: q.width,
        height: q.height,
        depth: q.depth,
        array_length: q.array_length,
        level: q.options,
        planes: q.flags,
    };
    let at = |level: MetalLevel| level.texture_level_info(&query);
    let oldest = at(MetalLevel::MacOS26_7);
    if !MetalLevel::ALL.into_iter().all(|level| at(level) == oldest) {
        crate::platform::answering_level(sel, || {
            format!(
                "Metal's layout of a level of pixel format {}",
                q.pixel_format
            )
        });
    }
    let [
        field_0,
        field_8,
        field_10,
        field_18,
        field_20,
        field_28,
        field_30,
        aligned_height,
        field_40,
        bytes_per_pixel,
        bytes_per_row,
        field_58,
        field_60,
    ] = device.texture_level_info(&query).words();
    TextureLevelInfo {
        field_0,
        field_8,
        field_10,
        field_18,
        field_20,
        field_28,
        field_30,
        aligned_height,
        field_40,
        bytes_per_pixel,
        bytes_per_row,
        field_58,
        field_60,
    }
}

/// Metal's name for `raw`, its description's at the devices' Metal level
/// (`MTLPixelFormatInvalid` for a value it describes no format for), for diagnostics: the levels
/// the runtime measured name a value alike but where one describes it and another does not (no
/// value at the runtime's levels), where `sel` asks the level ([`crate::platform::metal_level`]).
/// It never dies.
pub(crate) fn name(sel: &'static str, raw: u64) -> &'static str {
    read_at(|| crate::platform::metal_level(sel), raw, |info| info.name)
}

/// What this platform's own rules read of Metal's description of a format ([`block`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    /// Whether the format is one of blocks: flags bit 10 ([`COMPRESSED`]).
    compressed: bool,
    /// The description's four words.
    words: [u64; 4],
}

impl Block {
    /// Whether the format is one of blocks: flags bit 10 ([`COMPRESSED`]), the compressed formats
    /// and the 4:2:2 `GBGR422` and `BGRG422`.
    pub(crate) fn compressed(self) -> bool {
        self.compressed
    }

    /// A block's bytes (the first word): a texel's for a format not of blocks, 0 for a value Metal
    /// describes no format for.
    pub(crate) fn bytes(self) -> u64 {
        let [bytes, ..] = self.words;
        bytes
    }

    /// The block's width and height in texels (the second and third words) for a format of
    /// blocks; one texel for any other.
    pub(crate) fn extent(self) -> [u64; 2] {
        let [_, width, height, _] = self.words;
        match self.compressed {
            true => [width, height],
            false => [1, 1],
        }
    }
}

/// What this platform's own rules read of Metal's description of `raw` (a fill's texel,
/// [`crate::gpu::fill`]; a texture's bytes by Apple's layout, [`crate::gpu::metal_space`]; a CPU
/// write's unit, [`crate::gpu::resource`]): macOS 26.7's, where the rules were measured. The
/// levels the runtime measured describe it alike but where one describes `raw` and another does
/// not (no value at the runtime's levels): there `sel` asks the level
/// ([`crate::platform::answering_level`]) and, at any but macOS 26.7, which describes none, reads
/// that level's description, relayed naming
/// `what` (`a fill of`, ...), as nothing measured the rule at a release that describes the
/// format, and dies in strict mode ([`crate::answer::Inexact::MetalLevel`]).
pub(crate) fn block(sel: &'static str, raw: u64, what: &str) -> Block {
    let level = || crate::platform::answering_level(sel, || format!("{what} pixel format {raw}"));
    block_at(level, sel, raw, what)
}

/// [`block`] at the level `level` answers, which is asked only where the levels differ.
fn block_at(level: impl FnOnce() -> MetalLevel, sel: &'static str, raw: u64, what: &str) -> Block {
    let rules = MetalLevel::MacOS26_7;
    let level = || {
        let level = level();
        if level != rules {
            crate::answer::inexact(sel, crate::answer::Inexact::MetalLevel, || {
                format!(
                    "{what} pixel format {raw}, which Metal at {level} describes {}: this \
                     platform's rule for it was measured at {rules}, which describes {}",
                    describes(level, raw),
                    describes(rules, raw),
                )
            });
        }
        level
    };
    read_at(level, raw, |info| Block {
        compressed: info.flags & COMPRESSED != 0,
        words: info.words,
    })
}

/// "a format for" or "no format for", as `level` describes `raw`.
fn describes(level: MetalLevel, raw: u64) -> &'static str {
    match level.pixel_format_info(raw).describes_a_format() {
        true => "a format for",
        false => "no format for",
    }
}

/// `read` of Metal's description of `raw` at macOS 26.7, where every level the runtime measured
/// reads it alike ([`alike`]), and else at the level `level` answers, which is asked only then.
fn read_at<T: PartialEq>(
    level: impl FnOnce() -> MetalLevel,
    raw: u64,
    read: impl Fn(rt::PixelFormatInfo) -> T,
) -> T {
    match alike(raw, &read) {
        true => read(MetalLevel::MacOS26_7.pixel_format_info(raw)),
        false => read(level().pixel_format_info(raw)),
    }
}

/// Whether every Metal level the runtime measured gives `raw` a description of which `read` reads
/// the same.
fn alike<T: PartialEq>(raw: u64, read: impl Fn(rt::PixelFormatInfo) -> T) -> bool {
    let at = |level: MetalLevel| read(level.pixel_format_info(raw));
    let oldest = at(MetalLevel::MacOS26_7);
    MetalLevel::ALL.into_iter().all(|level| at(level) == oldest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The selector a description is asked by here.
    const INFO: &str = "MTLPixelFormatGetInfoForDevice()";

    /// Every value a level the runtime measured describes, from 0 to 2047: 264 at macOS 26.7.
    fn described_by(level: MetalLevel) -> Vec<u64> {
        (0..2048)
            .filter(|&raw| level.pixel_format_info(raw).describes_a_format())
            .collect()
    }

    /// The runtime has a format of every value Metal describes, at every level it measured, so a
    /// value this platform keeps raw ([`crate::gpu::held_format`]) is one Metal describes no
    /// format for: the death of a texture, view or state of one is Metal's assertion
    /// ([`crate::gpu::desc::lacked_as`])
    #[test]
    fn the_runtime_has_every_format_metal_describes() {
        assert_eq!(described_by(MetalLevel::MacOS26_7).len(), 264);
        for level in MetalLevel::ALL {
            for raw in described_by(level) {
                assert!(crate::gpu::runtime_pixel_format(raw).is_some(), "{raw}");
                assert!(described_at(|| level, raw), "{level} {raw}");
            }
        }
    }

    /// The runtime's row of macOS 26.7, at every value from 0 to 2047, is the measured host's
    /// Metal's: its setter takes what that level knows (the formats it describes,
    /// `MTLPixelFormatInvalid`, 255, 262 and 630 to 650, measured), and the level describes what
    /// its description describes (the Metal platform's on this host,
    /// `descriptions_are_the_metal_platforms_here`), which leaves out the two formats no Apple
    /// GPU has, `Depth24Unorm_Stencil8` (255) and `X24_Stencil8` (262); the levels differ only at
    /// the three-channel formats, where alone the devices' level is asked, and describe every
    /// value alike (`level_decides`)
    #[test]
    fn the_measured_host_is_the_runtimes_level_of_macos_26_7() {
        let described = described_by(MetalLevel::MacOS26_7);
        let measured = |raw| described.contains(&raw);
        let level = || MetalLevel::MacOS26_7;
        let unasked = || -> MetalLevel { panic!("the level is asked") };
        for raw in 0..2048 {
            let taken = measured(raw) || matches!(raw, 0 | 255 | 262 | 630..=650);
            let knows = |level: MetalLevel, pf| level.knows(pf);
            assert_eq!(at_level(level, raw, knows), taken, "{raw}");
            assert_eq!(described_at(level, raw), measured(raw), "{raw}");
            assert!(!level_decides(raw), "{raw}");
            if !matches!(raw, 45..=48 | 95..=100 | 120..=122) {
                at_level(unasked, raw, knows);
                described_at(unasked, raw);
            }
        }
    }

    /// The extents a level's layout is compared at: zeros, which Metal answers as 1, sizes that
    /// round to blocks and that do not, and sizes whose block counts or bytes wrap.
    const EXTENTS: [u64; 12] = [
        0,
        1,
        3,
        4,
        5,
        16,
        17,
        100,
        4096,
        16384,
        (1 << 62) + 3,
        u64::MAX,
    ];

    /// Metal's layouts where a level's block count wraps, as the M3 Max (macOS 26.7) answered
    /// them: Metal counts an extent's blocks as `(extent + block - 1) / block` with an add that
    /// wraps at 2^64, so a width or height past 2^64 - block has no blocks (`u64::MAX` has none 4
    /// wide), and the extent they cover and the bytes of a row, an image and the level follow from
    /// the count. The runtime answers each so at macOS 26.7 (`levels_are_the_metal_platforms_here`
    /// compares every format so on the device). Core never asks one: its extents are the guest's
    /// 32-bit width and height
    #[test]
    fn wrapped_block_counts_are_metals() {
        const MAX: u64 = u64::MAX;
        const PAST_2_62: u64 = (1 << 62) + 3;
        // (format, width, height, Metal's 13 words): BC1_RGBA (4 by 4 blocks of 8 bytes),
        // ASTC_12x10_LDR (12 by 10 of 16) and GBGR422 (2 by 1 of 4).
        let measured: [(u64, u64, u64, [u64; 13]); 4] = [
            (130, MAX, 17, [MAX, 17, 1, 0, 5, 1, 0, 20, 1, 8, 0, 0, 0]),
            (130, 17, MAX, [17, MAX, 1, 5, 0, 1, 20, 0, 1, 8, 40, 0, 0]),
            (
                217,
                PAST_2_62,
                MAX,
                [
                    PAST_2_62,
                    MAX,
                    1,
                    384_307_168_202_282_326,
                    0,
                    1,
                    4_611_686_018_427_387_912,
                    0,
                    1,
                    16,
                    6_148_914_691_236_517_216,
                    0,
                    0,
                ],
            ),
            (240, MAX, 5, [MAX, 5, 1, 0, 5, 1, 0, 5, 1, 4, 0, 0, 0]),
        ];
        for (format, width, height, metal) in measured {
            let q = rt::TextureLevelQuery::plane_2d(format, width, height);
            let runtime = MetalLevel::MacOS26_7.texture_level_info(&q);
            assert_eq!(runtime.words(), metal, "{format}, {width} by {height}");
        }
    }

    /// Where the levels the runtime measured describe a value otherwise, as the module's docs say:
    /// in the flags (where a description dies on a host whose release the runtime has not
    /// measured) of 586, 587, 623, 628 and 629 alone, none of them a compressed bit; in the cast
    /// class at 175 of the 264 formats both describe; nowhere in a level's layout, in what this
    /// platform's rules read ([`block`]) or in the name ([`level_decides`])
    #[test]
    fn descriptions_differ_by_level_only_where_the_runtime_measured_it() {
        let values: Vec<u64> = (0..2048)
            .chain([9999, u64::from(u32::MAX) + 70, u64::MAX])
            .collect();
        let differ = |read: &dyn Fn(rt::PixelFormatInfo) -> (u64, u64, [u64; 4])| -> Vec<u64> {
            values
                .iter()
                .copied()
                .filter(|&raw| !alike(raw, read))
                .collect()
        };
        let guest = differ(&|i| (i.flags, 0, i.words));
        assert_eq!(guest, [586, 587, 623, 628, 629]);
        assert!(differ(&|i| (i.flags & COMPRESSED, 0, i.words)).is_empty());
        assert_eq!(differ(&|i| (0, i.cast_class, [0; 4])).len(), 175);
        for &raw in &values {
            assert_eq!(!alike(raw, |i| i.name), level_decides(raw), "{raw}");
            let layouts = EXTENTS.map(|w| {
                let q = rt::TextureLevelQuery::plane_2d(raw, w, 9);
                MetalLevel::ALL.map(|l| l.texture_level_info(&q))
            });
            let alike = layouts.iter().all(|[a, b]| a == b);
            assert_eq!(!alike, level_decides(raw), "{raw}");
        }
    }

    /// What this platform's rules read of a description is macOS 26.7's, asked of no level where
    /// every level reads it alike, as at 630 to 645, which no level describes (an Apple M4 under
    /// 27.0.1 does): of no format, named `MTLPixelFormatInvalid`
    #[test]
    fn rules_read_macos_26_7s_description() {
        let unasked = || -> MetalLevel { panic!("the level is asked") };
        let bgra = block_at(unasked, INFO, 80, "a fill of");
        assert_eq!(
            (bgra.compressed(), bgra.bytes(), bgra.extent()),
            (false, 4, [1, 1])
        );
        let astc = block_at(unasked, INFO, 217, "a fill of");
        assert_eq!(
            (astc.compressed(), astc.bytes(), astc.extent()),
            (true, 16, [12, 10])
        );
        for raw in 630..=645 {
            let none = block_at(unasked, INFO, raw, "a fill of");
            assert_eq!((none.bytes(), none.extent()), (0, [1, 1]), "{raw}");
            assert_eq!(
                read_at(unasked, raw, |i| i.name),
                "MTLPixelFormatInvalid",
                "{raw}"
            );
        }
        assert_eq!(read_at(unasked, 80, |i| i.name), "MTLPixelFormatBGRA8Unorm");
    }

    #[cfg(target_os = "macos")]
    pub(crate) mod metal_tests;
}
