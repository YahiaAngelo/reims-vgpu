//! The guest's cursor drawn into a window frame where the window draws it itself, the guest moving
//! it on its own ([`DrawnCursor`], `reims_vgpu_window::handle`'s module doc): the glyph over the
//! frame the display drew, by its alpha, at the guest's cursor position the window last read.
//!
//! The glyph is a second draw in the display's own render pass, after the frame's
//! ([`DisplayCore::encode_current_frame_with`]), so the frame's target is stored once, with no
//! pass that loads it again: the display's strip, a quad over the whole viewport, drawn over a
//! viewport that is the glyph's rectangle in the target, its hot spot at the cursor's position, a
//! glyph pixel to a guest pixel at the image's scale, with a scissor that clips it to the image.
//! It draws with the display's overlay pipeline for the target's format
//! ([`DisplayCore::overlay_pipeline`]): the display's vertex function and
//! `displayPresentFragment`, which samples the glyph as the frame is sampled, filtered where the
//! image scales it, blending over by alpha (`RenderPipelineDesc::blend_over`). The guest's alpha
//! is straight, and the texture holds the glyph premultiplied, so that the filter, where the
//! image scales the glyph, does not darken its edges with the colour of its transparent pixels:
//! at one texel a pixel, each pixel is the glyph's colour over the frame's by its alpha, as with
//! straight alpha. The glyph's colours are drawn as the guest gives them, with no gamma table or
//! colour matrix, as the host's cursor shows them over the image.
//!
//! **What the guest reads.** The draw makes no buffer: it binds the display's. The pipeline is
//! the display's, kept in its cache with its own and made of its functions, so it lives as they
//! do. The glyph's texture is the sink's platform's own ([`OwnTexture`]): on the Vulkan platform
//! one that takes no resource ID and no range of Metal's space, so that no address or ID the
//! guest reads moves for it; on Metal a plain texture, a resource of Metal's like the layer's
//! drawables the window already draws into. The texture is made again only for a new glyph.
//!
//! **Before the frame.** The pipeline and the texture are made before the display encodes the
//! frame ([`CursorPass::prepare`]), from which the frame counts as in flight, which a guest's
//! present waits on. What the platform refuses of them leaves the frame without the guest's
//! cursor, logged once; the host's cursor stays hidden over the image while the guest leads, so
//! neither shows. An encode the platform refuses dies at its send, as the display's own does; the
//! draw's values are ones the platforms take: a scissor inside the target, and a viewport inside
//! the least bounds Vulkan sets every device (the runtime hands a viewport to Vulkan as it is),
//! [`VIEWPORT_BOUNDS`] and [`VIEWPORT_DIMENSION`]; a glyph that would reach past them is not drawn
//! (only near the far edge of a window over 8000 pixels across). Nothing here reaches the guest
//! or the C API: the cursor is in the window's pixels alone.

use std::sync::Arc;

use reims_vgpu_core::display::DisplayCore;
use reims_vgpu_core::display::pipeline::OverlayDraw;
use reims_vgpu_core::platform::resource::Texture as _;
use reims_vgpu_core::platform::types::{
    MtlOrigin, MtlRegion, MtlScissorRect, MtlSize, MtlViewport,
};
use reims_vgpu_core::platform::values::TextureDesc;
use reims_vgpu_core::platform::{Handle as _, HostRange, Platform};
use reims_vgpu_window::{DrawnCursor, Glyph, PixelRect};

use crate::display::no_catch;

/// The least `viewportBoundsRange` Vulkan sets every device: a viewport's edges lie inside it.
const VIEWPORT_BOUNDS: (f64, f64) = (-8192.0, 8191.0);
/// The least `maxViewportDimensions` Vulkan sets every device: a viewport's width and height.
const VIEWPORT_DIMENSION: f64 = 4096.0;

/// How a sink's platform makes the glyph's texture from its descriptor: a texture the guest never
/// reaches (the module doc), or `None` where the platform makes none.
pub(crate) type OwnTexture<P> = fn(
    &<P as Platform>::Device,
    &<P as Platform>::TextureDescriptor,
) -> Option<<P as Platform>::Texture>;

/// What draws the guest's cursor over a window's frames, kept from one frame to the next.
pub(crate) struct CursorPass<P: Platform> {
    /// How the glyph's texture is made.
    own_texture: OwnTexture<P>,
    /// The glyph last drawn, and its texture.
    glyph: Option<(Arc<Glyph>, P::Texture)>,
    /// A refusal of the platform's has been logged.
    refusal_logged: bool,
}

/// One frame's cursor, made ready before the display encodes the frame
/// ([`CursorPass::prepare`]), which draws it ([`Prepared::draw`]).
pub(crate) struct Prepared<P: Platform> {
    /// The display's overlay pipeline for the target's format.
    pipeline: P::RenderPipeline,
    /// The glyph's texture.
    texture: P::Texture,
    /// The glyph's rectangle in the target, and its clip to the image ([`placement`]).
    viewport: MtlViewport,
    /// The scissor.
    scissor: MtlScissorRect,
}

impl<P: Platform> CursorPass<P> {
    /// A pass that has made nothing yet, whose glyph textures `own_texture` makes.
    pub(crate) fn new(own_texture: OwnTexture<P>) -> CursorPass<P> {
        CursorPass {
            own_texture,
            glyph: None,
            refusal_logged: false,
        }
    }

    /// What drawing `cursor` over the frame `display` draws into `target` over `region` binds:
    /// the display's overlay pipeline for the target's format, made the first time
    /// ([`DisplayCore::overlay_pipeline`]), the glyph's texture, made again only for a new
    /// glyph, and where it draws ([`placement`]). Asked before the frame is encoded (the module
    /// doc). `None` where nothing would show, and where the platform refuses the pipeline or the
    /// texture (logged once).
    pub(crate) fn prepare(
        &mut self,
        display: &DisplayCore<P>,
        target: &P::Texture,
        region: PixelRect,
        cursor: &DrawnCursor,
    ) -> Option<Prepared<P>> {
        let (viewport, scissor) = placement(region, (target.width(), target.height()), cursor)?;
        let Some(pipeline) = display.overlay_pipeline(target.pixel_format()) else {
            return self.refused("the pipeline");
        };
        let texture = match &self.glyph {
            Some((drawn, texture)) if Arc::ptr_eq(drawn, &cursor.glyph) => texture.clone_ref(),
            _ => {
                let made = display.mtl_device().and_then(|device| {
                    glyph_texture(display.platform(), device, self.own_texture, &cursor.glyph)
                });
                let Some(texture) = made else {
                    return self.refused("a texture for the glyph");
                };
                self.glyph = Some((Arc::clone(&cursor.glyph), texture.clone_ref()));
                texture
            }
        };
        Some(Prepared {
            pipeline,
            texture,
            viewport,
            scissor,
        })
    }

    /// Logs, the first time, that the platform refused `what`: the cursor is not drawn.
    fn refused<T>(&mut self, what: &str) -> Option<T> {
        if !std::mem::replace(&mut self.refusal_logged, true) {
            log::warn!(
                target: "reims_vgpu::window",
                "the window draws no guest cursor: the platform refused {what}"
            );
        }
        None
    }
}

impl<P: Platform> Prepared<P> {
    /// The draw the display makes of the cursor in its pass, after the frame's (the module doc).
    pub(crate) fn draw(&self) -> OverlayDraw<'_, P> {
        OverlayDraw {
            pipeline: &self.pipeline,
            texture: &self.texture,
            viewport: self.viewport,
            scissor: self.scissor,
        }
    }
}

/// Where `cursor` draws over `region` of a target of `size` pixels: the viewport, the glyph's
/// rectangle in the target, its hot spot at the cursor's position in the mode's pixels, a glyph
/// pixel to a guest pixel at the image's scale; and the scissor, the target's pixels that
/// rectangle touches inside the image and the target. The viewport may reach past the target,
/// which only the scissor clips. `None` where nothing would show: an empty mode, glyph or
/// region, or a glyph wholly off the image; and where the viewport would reach past
/// [`VIEWPORT_BOUNDS`] or be larger than [`VIEWPORT_DIMENSION`] (the module doc).
fn placement(
    region: PixelRect,
    size: (u64, u64),
    cursor: &DrawnCursor,
) -> Option<(MtlViewport, MtlScissorRect)> {
    let (mode, glyph) = (cursor.mode, &cursor.glyph);
    if mode.0 == 0 || mode.1 == 0 || glyph.width == 0 || glyph.height == 0 {
        return None;
    }
    #[expect(clippy::as_conversions, reason = "u64 to f64 rounds to nearest")]
    let [x, y, width, height, target_width, target_height] = [
        region.x,
        region.y,
        region.width,
        region.height,
        size.0,
        size.1,
    ]
    .map(|v| v as f64);
    let (per_x, per_y) = (width / f64::from(mode.0), height / f64::from(mode.1));
    let left = x + (f64::from(cursor.at.0) - f64::from(glyph.hot_spot.0)) * per_x;
    let top = y + (f64::from(cursor.at.1) - f64::from(glyph.hot_spot.1)) * per_y;
    let (glyph_width, glyph_height) = (
        f64::from(glyph.width) * per_x,
        f64::from(glyph.height) * per_y,
    );
    let (x0, y0) = (left.floor().max(x), top.floor().max(y));
    let x1 = (left + glyph_width).ceil().min(x + width).min(target_width);
    let y1 = (top + glyph_height)
        .ceil()
        .min(y + height)
        .min(target_height);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let (low, high) = VIEWPORT_BOUNDS;
    let within = |start: f64, length: f64| {
        start >= low && start + length <= high && length <= VIEWPORT_DIMENSION
    };
    if !(within(left, glyph_width) && within(top, glyph_height)) {
        return None;
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "whole pixels inside the target, which u64 holds"
    )]
    let [sx, sy, sw, sh] = [x0, y0, x1 - x0, y1 - y0].map(|v| v as u64);
    let viewport = MtlViewport {
        origin_x: left,
        origin_y: top,
        width: glyph_width,
        height: glyph_height,
        znear: 0.0,
        zfar: 1.0,
    };
    let scissor = MtlScissorRect {
        x: sx,
        y: sy,
        width: sw,
        height: sh,
    };
    Some((viewport, scissor))
}

/// `glyph`'s pixels in a texture `own_texture` makes, RGBA8, premultiplied (module doc), read by
/// shaders, a row its bytes do not reach transparent; `None` for a glyph with no pixels, or
/// where the platform makes no texture.
fn glyph_texture<P: Platform>(
    platform: &Arc<P>,
    device: &P::Device,
    own_texture: OwnTexture<P>,
    glyph: &Glyph,
) -> Option<P::Texture> {
    let (width, height) = (u64::from(glyph.width), u64::from(glyph.height));
    let row = width.checked_mul(4)?;
    let length = usize::try_from(row.checked_mul(height)?).ok()?;
    if length == 0 {
        return None;
    }
    let mut bytes = glyph.rgba.get(..length).unwrap_or(&glyph.rgba).to_vec();
    bytes.resize(length, 0);
    premultiply(&mut bytes);
    let descriptor = platform.texture_descriptor(&TextureDesc {
        width,
        height,
        ..TextureDesc::default()
    });
    let texture = own_texture(device, &descriptor)?;
    no_catch(texture.replace_region(
        MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(width, height, 1)),
        0,
        0,
        HostRange::of_slice(&bytes),
        row,
        row * height,
    ));
    Some(texture)
}

/// Premultiplies RGBA `pixels` by their alpha, each channel rounded to nearest.
pub(super) fn premultiply(pixels: &mut [u8]) {
    for [red, green, blue, alpha] in pixels.as_chunks_mut::<4>().0 {
        let alpha = u16::from(*alpha);
        for channel in [red, green, blue] {
            let scaled = (u16::from(*channel) * alpha + 127) / 255;
            *channel = u8::try_from(scaled).unwrap_or(u8::MAX);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cursor of a `width` by `height` glyph with its hot spot at `hot_spot`, at `at` in `mode`.
    fn cursor(
        (width, height): (u32, u32),
        hot_spot: (u16, u16),
        mode: (u16, u16),
        at: (u16, u16),
    ) -> DrawnCursor {
        DrawnCursor {
            glyph: Arc::new(Glyph {
                width,
                height,
                hot_spot,
                rgba: vec![0xff; usize::try_from(width * height * 4).unwrap()],
            }),
            mode,
            at,
        }
    }

    /// A region of the target.
    fn rect(x: u64, y: u64, width: u64, height: u64) -> PixelRect {
        PixelRect {
            x,
            y,
            width,
            height,
        }
    }

    /// The viewport puts the glyph's hot spot at the guest's cursor position, a glyph pixel to a
    /// guest pixel at the image's scale, in a letterboxed image too, and the scissor is the
    /// glyph's rectangle where it lies inside the image
    #[test]
    fn the_viewport_puts_the_hot_spot_at_the_cursor() {
        let c = cursor((10, 5), (10, 0), (100, 50), (60, 20));
        let (v, s) = placement(rect(0, 0, 100, 50), (100, 50), &c).unwrap();
        assert_eq!(
            (v.origin_x, v.origin_y, v.width, v.height),
            (50.0, 20.0, 10.0, 5.0)
        );
        assert_eq!((s.x, s.y, s.width, s.height), (50, 20, 10, 5));
        // Twice the size, 30 pixels from the left and 10 from the top.
        let (v, s) = placement(rect(30, 10, 200, 100), (260, 120), &c).unwrap();
        assert_eq!(
            (v.origin_x, v.origin_y, v.width, v.height),
            (130.0, 50.0, 20.0, 10.0)
        );
        assert_eq!((s.x, s.y, s.width, s.height), (130, 50, 20, 10));
        // At a scale of 1.5 the rectangle's edges fall inside pixels, which the scissor takes.
        let (v, s) = placement(rect(0, 0, 150, 75), (150, 75), &c).unwrap();
        assert_eq!((v.origin_x, v.width), (75.0, 15.0));
        assert_eq!((v.origin_y, v.height), (30.0, 7.5));
        assert_eq!((s.x, s.y, s.width, s.height), (75, 30, 15, 8));
    }

    /// A glyph partly off the image keeps its whole viewport, past the target where it reaches
    /// there, and the scissor clips it to the image; one wholly off it, an empty glyph and an
    /// empty mode draw nothing
    #[test]
    fn the_scissor_clips_the_glyph_to_the_image() {
        let region = rect(20, 10, 100, 100);
        let size = (140, 120);
        let g = (16, 16);
        let (v, s) = placement(region, size, &cursor(g, (4, 4), (100, 100), (0, 0))).unwrap();
        assert_eq!((v.origin_x, v.origin_y, v.width), (16.0, 6.0, 16.0));
        assert_eq!((s.x, s.y, s.width, s.height), (20, 10, 12, 12));
        let (v, s) = placement(region, size, &cursor(g, (0, 0), (100, 100), (95, 95))).unwrap();
        assert_eq!((v.origin_x, v.origin_y), (115.0, 105.0));
        assert_eq!((s.x, s.y, s.width, s.height), (115, 105, 5, 5));
        let at_edge = cursor(g, (8, 8), (100, 100), (0, 0));
        let (v, s) = placement(rect(0, 0, 100, 100), (100, 100), &at_edge).unwrap();
        assert_eq!((v.origin_x, v.origin_y), (-8.0, -8.0));
        assert_eq!((s.x, s.y, s.width, s.height), (0, 0, 8, 8));
        let none = |c: &DrawnCursor| placement(region, size, c).is_none();
        assert!(none(&cursor(g, (0, 0), (100, 100), (100, 50))));
        assert!(none(&cursor(g, (0, 0), (100, 100), (50, 100))));
        assert!(none(&cursor(g, (0, 0), (100, 100), (u16::MAX, u16::MAX))));
        assert!(none(&cursor(g, (20, 20), (100, 100), (0, 0))));
        assert!(none(&cursor((0, 16), (0, 0), (100, 100), (50, 50))));
        assert!(none(&cursor(g, (0, 0), (0, 100), (50, 50))));
        assert!(
            placement(
                rect(20, 10, 0, 100),
                size,
                &cursor(g, (0, 0), (100, 100), (5, 5))
            )
            .is_none()
        );
    }

    /// A glyph whose viewport would reach past Vulkan's least viewport bounds, or be larger than
    /// its least viewport size, is not drawn, though it shows in the image
    #[test]
    fn a_viewport_past_vulkan_s_least_bounds_draws_nothing() {
        let wide = rect(0, 0, 8192, 100);
        let size = (8192, 100);
        let g = (16, 16);
        assert!(placement(wide, size, &cursor(g, (0, 0), (8192, 100), (8170, 10))).is_some());
        assert!(placement(wide, size, &cursor(g, (0, 0), (8192, 100), (8180, 10))).is_none());
        // A glyph of 256 pixels at 17 pixels to a guest pixel is 4352 wide.
        let scaled = rect(0, 0, 1700, 1700);
        let big = cursor((256, 1), (0, 0), (100, 100), (0, 0));
        assert!(placement(scaled, (1700, 1700), &big).is_none());
        let fits = cursor((240, 1), (0, 0), (100, 100), (0, 0));
        assert!(placement(scaled, (1700, 1700), &fits).is_some());
    }

    /// Premultiplying scales each colour channel by the alpha, to nearest, and keeps the alpha
    #[test]
    fn premultiplying_scales_the_colour_by_the_alpha() {
        let mut pixels = [
            0xff, 0x80, 0x00, 0xff, 0xff, 0x80, 0x10, 0x80, 0xff, 0xff, 0xff, 0x00,
        ];
        premultiply(&mut pixels);
        assert_eq!(
            pixels,
            [
                0xff, 0x80, 0x00, 0xff, 0x80, 0x40, 0x08, 0x80, 0x00, 0x00, 0x00, 0x00
            ]
        );
    }
}
