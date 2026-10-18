//! ASTC on a device without it (vphone-linux).
//!
//! An iOS guest's UI is drawn from ASTC textures (asset catalogs), which Apple's GPUs decode and
//! a desktop GPU's Vulkan driver often lacks (`textureCompressionASTC_LDR` false: NVIDIA). Where
//! the runtime refuses a texture of an ASTC format, it is made of `RGBA8Unorm` (`_sRGB` for an
//! sRGB format) at the same texel extent instead, and marked with its block footprint
//! ([`metal2vulkan_runtime::Texture::set_astc_blocks`]). What the guest writes into it as blocks
//! (`replaceRegion:`, a blit from a buffer) is decoded here on the CPU and written as texels.
//!
//! HDR blocks are decoded as LDR (clamped), and a texture over a buffer's bytes or an IOSurface
//! keeps its refusal: no decode can stand between such a texture and the memory it reads.

use metal2vulkan::metal::PixelFormat as P;
use metal2vulkan_runtime as rt;

/// The block footprint of an ASTC format and whether it is an sRGB one; `None` for any other.
pub fn blocks(pf: P) -> Option<([u32; 2], bool)> {
    let fp = match pf {
        P::ASTC4x4LDR | P::ASTC4x4SRGB | P::ASTC4x4HDR => [4, 4],
        P::ASTC5x4LDR | P::ASTC5x4SRGB | P::ASTC5x4HDR => [5, 4],
        P::ASTC5x5LDR | P::ASTC5x5SRGB | P::ASTC5x5HDR => [5, 5],
        P::ASTC6x5LDR | P::ASTC6x5SRGB | P::ASTC6x5HDR => [6, 5],
        P::ASTC6x6LDR | P::ASTC6x6SRGB | P::ASTC6x6HDR => [6, 6],
        P::ASTC8x5LDR | P::ASTC8x5SRGB | P::ASTC8x5HDR => [8, 5],
        P::ASTC8x6LDR | P::ASTC8x6SRGB | P::ASTC8x6HDR => [8, 6],
        P::ASTC8x8LDR | P::ASTC8x8SRGB | P::ASTC8x8HDR => [8, 8],
        P::ASTC10x5LDR | P::ASTC10x5SRGB | P::ASTC10x5HDR => [10, 5],
        P::ASTC10x6LDR | P::ASTC10x6SRGB | P::ASTC10x6HDR => [10, 6],
        P::ASTC10x8LDR | P::ASTC10x8SRGB | P::ASTC10x8HDR => [10, 8],
        P::ASTC10x10LDR | P::ASTC10x10SRGB | P::ASTC10x10HDR => [10, 10],
        P::ASTC12x10LDR | P::ASTC12x10SRGB | P::ASTC12x10HDR => [12, 10],
        P::ASTC12x12LDR | P::ASTC12x12SRGB | P::ASTC12x12HDR => [12, 12],
        _ => return None,
    };
    let srgb = matches!(
        pf,
        P::ASTC4x4SRGB
            | P::ASTC5x4SRGB
            | P::ASTC5x5SRGB
            | P::ASTC6x5SRGB
            | P::ASTC6x6SRGB
            | P::ASTC8x5SRGB
            | P::ASTC8x6SRGB
            | P::ASTC8x8SRGB
            | P::ASTC10x5SRGB
            | P::ASTC10x6SRGB
            | P::ASTC10x8SRGB
            | P::ASTC10x10SRGB
            | P::ASTC12x10SRGB
            | P::ASTC12x12SRGB
    );
    Some((fp, srgb))
}

/// The descriptor of the RGBA8 texture that stands for an ASTC one, and the block footprint;
/// `None` for a descriptor of another format.
pub fn substitute(desc: &rt::TextureDescriptor) -> Option<(rt::TextureDescriptor, [u32; 2])> {
    let (fp, srgb) = blocks(desc.pixel_format)?;
    let mut d = *desc;
    d.pixel_format = if srgb {
        P::RGBA8UnormSRGB
    } else {
        P::RGBA8Unorm
    };
    Some((d, fp))
}

/// Whether `e`, the runtime's refusal of a texture of `pf`, is one an RGBA8 stand-in answers:
/// any refusal of an ASTC format (the runtime names the format feature it lacks).
pub fn refused_astc(pf: P) -> bool {
    blocks(pf).is_some()
}

/// Logs the first stand-ins made, once each kind.
pub fn note_made(pf: P, width: u64, height: u64) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        log::info!(
            "ASTC texture {pf:?} {width}x{height} held as RGBA8 texels on a device without ASTC; \
             its blocks are decoded on the CPU"
        );
    }
}

/// The texels of a region `width` x `height` texels of ASTC blocks of footprint `fp`, the blocks
/// starting at `src` with rows of blocks `bytes_per_row` apart: tightly packed RGBA8 rows
/// (`width * 4` bytes each). Bytes past `src` read as zero blocks.
pub fn decode(src: &[u8], bytes_per_row: usize, fp: [u32; 2], width: u32, height: u32) -> Vec<u8> {
    let (bw, bh) = (fp[0], fp[1]);
    let blocks_x = width.div_ceil(bw);
    let blocks_y = height.div_ceil(bh);
    let footprint = astc_decode::Footprint::new(bw, bh);
    let row = width as usize * 4;
    let mut out = vec![0u8; row * height as usize];
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let at = by as usize * bytes_per_row + bx as usize * 16;
            let mut block = [0u8; 16];
            if let Some(bytes) = src.get(at..at + 16) {
                block.copy_from_slice(bytes);
            }
            astc_decode::astc_decode_block(&block, footprint, |x, y, px| {
                let (tx, ty) = (bx * bw + x, by * bh + y);
                if tx < width && ty < height {
                    let o = ty as usize * row + tx as usize * 4;
                    out[o..o + 4].copy_from_slice(&px);
                }
            });
        }
    }
    out
}

/// The ASTC format a stand-in of block footprint `fp` stands for, as the guest made it: the
/// LDR form, or the sRGB one for a stand-in of `RGBA8Unorm_sRGB` (an HDR texture answers LDR).
pub fn original(fp: [u32; 2], stand_in: P) -> P {
    let srgb = stand_in == P::RGBA8UnormSRGB;
    match (fp, srgb) {
        ([4, 4], false) => P::ASTC4x4LDR,
        ([4, 4], true) => P::ASTC4x4SRGB,
        ([5, 4], false) => P::ASTC5x4LDR,
        ([5, 4], true) => P::ASTC5x4SRGB,
        ([5, 5], false) => P::ASTC5x5LDR,
        ([5, 5], true) => P::ASTC5x5SRGB,
        ([6, 5], false) => P::ASTC6x5LDR,
        ([6, 5], true) => P::ASTC6x5SRGB,
        ([6, 6], false) => P::ASTC6x6LDR,
        ([6, 6], true) => P::ASTC6x6SRGB,
        ([8, 5], false) => P::ASTC8x5LDR,
        ([8, 5], true) => P::ASTC8x5SRGB,
        ([8, 6], false) => P::ASTC8x6LDR,
        ([8, 6], true) => P::ASTC8x6SRGB,
        ([8, 8], false) => P::ASTC8x8LDR,
        ([8, 8], true) => P::ASTC8x8SRGB,
        ([10, 5], false) => P::ASTC10x5LDR,
        ([10, 5], true) => P::ASTC10x5SRGB,
        ([10, 6], false) => P::ASTC10x6LDR,
        ([10, 6], true) => P::ASTC10x6SRGB,
        ([10, 8], false) => P::ASTC10x8LDR,
        ([10, 8], true) => P::ASTC10x8SRGB,
        ([10, 10], false) => P::ASTC10x10LDR,
        ([10, 10], true) => P::ASTC10x10SRGB,
        ([12, 10], false) => P::ASTC12x10LDR,
        ([12, 10], true) => P::ASTC12x10SRGB,
        (_, false) => P::ASTC12x12LDR,
        (_, true) => P::ASTC12x12SRGB,
    }
}

/// The blocks across and down of `raw`'s level `level`, of footprint `fp`.
fn level_blocks(raw: &rt::Texture, level: u64, fp: [u32; 2]) -> (usize, usize) {
    let w = (raw.width() >> level).max(1);
    let h = (raw.height() >> level).max(1);
    (w.div_ceil(u64::from(fp[0])) as usize, h.div_ceil(u64::from(fp[1])) as usize)
}

/// The block rect of a texel region: first block across and down, blocks across and down.
fn block_rect(fp: [u32; 2], origin: [u64; 2], size: [u64; 2]) -> [usize; 4] {
    let (bw, bh) = (u64::from(fp[0]), u64::from(fp[1]));
    [
        (origin[0] / bw) as usize,
        (origin[1] / bh) as usize,
        size[0].div_ceil(bw) as usize,
        size[1].div_ceil(bh) as usize,
    ]
}

/// Keeps the blocks the guest writes into a region of a stand-in, rows `bytes_per_row` apart
/// in `src`, in the texture's shadow ([`rt::Texture::astc_shadow`]), whose copies to a buffer
/// answer from it.
pub fn keep(
    raw: &rt::Texture,
    (level, slice): (u64, u64),
    fp: [u32; 2],
    (origin, size): ([u64; 2], [u64; 2]),
    src: &[u8],
    bytes_per_row: usize,
) {
    let (lx, ly) = level_blocks(raw, level, fp);
    let [x0, y0, nx, ny] = block_rect(fp, origin, size);
    raw.astc_shadow(level, slice, |shadow| {
        shadow.resize(lx * ly * 16, 0);
        for r in 0..ny.min(ly.saturating_sub(y0)) {
            let n = nx.min(lx.saturating_sub(x0)) * 16;
            let from = r * bytes_per_row;
            let to = ((y0 + r) * lx + x0) * 16;
            if let Some(bytes) = src.get(from..from + n) {
                shadow[to..to + n].copy_from_slice(bytes);
            }
        }
    });
}

/// The blocks kept of a region of a stand-in: rows of blocks tightly packed, and their length.
pub fn kept(
    raw: &rt::Texture,
    (level, slice): (u64, u64),
    fp: [u32; 2],
    (origin, size): ([u64; 2], [u64; 2]),
) -> (Vec<u8>, usize) {
    let (lx, ly) = level_blocks(raw, level, fp);
    let [x0, y0, nx, ny] = block_rect(fp, origin, size);
    let row = nx * 16;
    let mut out = vec![0u8; row * ny];
    raw.astc_shadow(level, slice, |shadow| {
        // Never written: zero blocks.
        if shadow.is_empty() {
            return;
        }
        for r in 0..ny.min(ly.saturating_sub(y0)) {
            let n = nx.min(lx.saturating_sub(x0)) * 16;
            let from = ((y0 + r) * lx + x0) * 16;
            out[r * row..r * row + n].copy_from_slice(&shadow[from..from + n]);
        }
    });
    (out, row)
}
