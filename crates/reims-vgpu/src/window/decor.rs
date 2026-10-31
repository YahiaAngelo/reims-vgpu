//! What a window draws around the guest's frame at its control's say
//! ([`Decoration`], `reims_vgpu_window::decor`): the frame turned a quarter, a half or three
//! quarters into the image's rectangle, and the button bar beside it. Only the Vulkan platform's
//! sink draws one (Linux's window, whose control socket asks for it); a window with none draws
//! as before, through the display's pass alone.
//!
//! **Two passes in the frame's command buffer.** Turned, the display draws its frame, and the
//! guest's cursor over it, into a texture of the window's own at the image's unturned size
//! ([`Prepared::frame_target`]), as it would into the window, and the window's pass
//! ([`DisplayCore::encode_window_pass`]) clears the swapchain's image to black and draws that
//! texture into the image's rectangle with the strip's texture coordinates turned
//! ([`Rotation::tex_coords`]), through the display's copy pipeline (no gamma table, no
//! blending: the frame's colours as the display drew them, its alpha too, as the display's pass
//! writes it into the window). Unturned, the display draws into the swapchain's image as always,
//! and the window's pass loads it. Either way the pass then draws the bar: its background, a
//! texel of [`BAR_BACKGROUND`] over the bar's rectangle through the copy pipeline, and each
//! button, its cell of the atlas, through the display's overlay pipeline, which blends it over
//! the background by its alpha, the atlas held premultiplied as the cursor's glyph is
//! (`super::cursor`). A button's cell is drawn by a quad of the whole atlas, scaled so that a
//! cell is the button's rectangle and placed so that the button's cell lies over it, clipped to
//! the button by the scissor: the strip's texture coordinates stay the display's. Linear
//! filtering at a scale above one blends a cell's edge texels half with their neighbours'.
//!
//! **What the guest reads.** Nothing here reaches it. The textures (the turned frame's, the
//! atlas's, the background's) and the buffer of the strip's positions and texture coordinates
//! are the platform's own ([`OwnTexture`], [`OwnBuffer`]): on the Vulkan platform outside the
//! Metal address space, with no resource ID. They are made before the display encodes the
//! frame ([`DecorPass::prepare`]), the buffer once, the turned frame's texture again for a new
//! size, the atlas's for a new atlas. What the platform refuses of them leaves the frame drawn as
//! a plain one, logged once. A button whose quad would reach past Vulkan's least viewport bounds
//! ([`VIEWPORT_BOUNDS`], [`VIEWPORT_DIMENSION`]) is not drawn: only a bar of many large buttons
//! at a high scale.

use std::sync::Arc;

use reims_vgpu_core::display::pipeline::WindowDraw;
use reims_vgpu_core::display::{DisplayCore, POS_DATA};
use reims_vgpu_core::platform::resource::Texture as _;
use reims_vgpu_core::platform::types::{
    MtlOrigin, MtlRegion, MtlScissorRect, MtlSize, MtlViewport,
};
use reims_vgpu_core::platform::values::TextureDesc;
use reims_vgpu_core::platform::{Handle as _, HostRange, Platform};
use reims_vgpu_window::decor::BAR_BACKGROUND;
use reims_vgpu_window::{Atlas, BarDraw, Decoration, PixelRect, Rotation};

use super::cursor::OwnTexture;
use crate::display::no_catch;

/// The least `viewportBoundsRange` Vulkan sets every device: a viewport's edges lie inside it.
const VIEWPORT_BOUNDS: (f64, f64) = (-8192.0, 8191.0);
/// The least `maxViewportDimensions` Vulkan sets every device: a viewport's width and height.
const VIEWPORT_DIMENSION: f64 = 4096.0;

/// `MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget`: the turned frame's texture is
/// drawn into, then sampled.
const TURNED_USAGE: u64 = 1 | 4;
/// `MTLResourceStorageModePrivate`: the GPU's alone.
const STORAGE_PRIVATE: u64 = 2 << 4;

/// Where the strip's blocks lie in the vertex buffer, in bytes: each at a multiple of 256, the
/// largest offset alignment Vulkan lets a device ask of a buffer's binding.
const BLOCK: u64 = 256;
/// The positions' block: the display's strip (`POS_DATA`).
const POSITIONS: u64 = 0;

/// The texture coordinates' block for `rotation`.
fn tex_coords_at(rotation: Rotation) -> u64 {
    let index = match rotation {
        Rotation::None => 1,
        Rotation::Quarter => 2,
        Rotation::Half => 3,
        Rotation::ThreeQuarters => 4,
    };
    index * BLOCK
}

/// The vertex buffer's bytes: the positions, then each rotation's texture coordinates, each in
/// its block.
fn vertex_bytes() -> Vec<u8> {
    let mut bytes = vec![0; usize::try_from(5 * BLOCK).unwrap_or(0)];
    let mut put = |at: u64, floats: &[f32]| {
        let at = usize::try_from(at).unwrap_or(0);
        for (i, f) in floats.iter().enumerate() {
            bytes[at + 4 * i..at + 4 * i + 4].copy_from_slice(&f.to_ne_bytes());
        }
    };
    put(POSITIONS, &POS_DATA);
    for rotation in [
        Rotation::None,
        Rotation::Quarter,
        Rotation::Half,
        Rotation::ThreeQuarters,
    ] {
        put(tex_coords_at(rotation), &rotation.tex_coords());
    }
    bytes
}

/// How a sink's platform makes a buffer of its own holding the bytes given: one the guest never
/// reaches (the module doc), or `None` where the platform makes none.
pub(crate) type OwnBuffer<P> =
    fn(&<P as Platform>::Device, &[u8]) -> Option<<P as Platform>::Buffer>;

/// What draws a window's decorations, kept from one frame to the next.
pub(crate) struct DecorPass<P: Platform> {
    /// How its textures are made.
    own_texture: OwnTexture<P>,
    /// How its buffer is made.
    own_buffer: OwnBuffer<P>,
    /// The strip's positions and texture coordinates, once made.
    vertices: Option<P::Buffer>,
    /// The turned frame's texture, and its size.
    turned: Option<((u64, u64), P::Texture)>,
    /// The atlas last drawn, and its texture.
    atlas: Option<(Arc<Atlas>, P::Texture)>,
    /// The bar's background texel.
    background: Option<P::Texture>,
    /// A refusal of the platform's has been logged.
    refusal_logged: bool,
}

/// One frame's decorations, made ready before the display encodes the frame
/// ([`DecorPass::prepare`]).
pub(crate) struct Prepared<P: Platform> {
    /// The display's copy pipeline for the target's format.
    copy: P::RenderPipeline,
    /// The strip's buffer.
    vertices: P::Buffer,
    /// The turned frame: its texture, the image's rectangle in the target, and the turn.
    turned: Option<(P::Texture, PixelRect, Rotation)>,
    /// The bar.
    bar: Option<PreparedBar<P>>,
}

/// The bar of one frame.
struct PreparedBar<P: Platform> {
    /// The display's overlay pipeline for the target's format.
    overlay: P::RenderPipeline,
    /// The background texel.
    background: P::Texture,
    /// The bar's rectangle, clipped to the target.
    rect: MtlScissorRect,
    /// The atlas.
    atlas: P::Texture,
    /// Each button's quad and clip.
    buttons: Vec<(MtlViewport, MtlScissorRect)>,
}

impl<P: Platform> DecorPass<P> {
    /// A pass that has made nothing yet.
    pub(crate) fn new(own_texture: OwnTexture<P>, own_buffer: OwnBuffer<P>) -> DecorPass<P> {
        DecorPass {
            own_texture,
            own_buffer,
            vertices: None,
            turned: None,
            atlas: None,
            background: None,
            refusal_logged: false,
        }
    }

    /// What drawing `decoration` around the frame `display` draws into `target` over `region`
    /// binds (the module doc). `None` where it draws nothing of its own, and where the platform
    /// refuses a pipeline, a texture or the buffer (logged once): the frame is then drawn plain.
    pub(crate) fn prepare(
        &mut self,
        display: &DisplayCore<P>,
        target: &P::Texture,
        region: PixelRect,
        decoration: &Decoration,
    ) -> Option<Prepared<P>> {
        if decoration.is_plain() {
            return None;
        }
        let format = target.pixel_format();
        let Some(copy) = display.copy_pipeline(format) else {
            return self.refused("the copy pipeline");
        };
        let device = display.mtl_device()?;
        let platform = display.platform();
        if self.vertices.is_none() {
            self.vertices = (self.own_buffer)(device, &vertex_bytes());
        }
        let Some(vertices) = self.vertices.as_ref().map(|b| b.clone_ref()) else {
            return self.refused("the strip's buffer");
        };
        let turned = match decoration.rotation {
            Rotation::None => None,
            rotation if region.width > 0 && region.height > 0 => {
                let size = rotation.shown((region.width, region.height));
                let texture = match &self.turned {
                    Some((made, texture)) if *made == size => texture.clone_ref(),
                    _ => {
                        let descriptor = platform.texture_descriptor(&TextureDesc {
                            pixel_format: format,
                            width: size.0,
                            height: size.1,
                            usage: TURNED_USAGE,
                            resource_options: Some(STORAGE_PRIVATE),
                            ..TextureDesc::default()
                        });
                        let Some(texture) = (self.own_texture)(device, &descriptor) else {
                            return self.refused("a texture for the turned frame");
                        };
                        self.turned = Some((size, texture.clone_ref()));
                        texture
                    }
                };
                Some((texture, region, rotation))
            }
            _ => None,
        };
        let size = (target.width(), target.height());
        let bar = match &decoration.bar {
            Some(bar) => Some(self.prepare_bar(display, device, format, size, bar)?),
            None => None,
        };
        Some(Prepared {
            copy,
            vertices,
            turned,
            bar,
        })
    }

    /// The bar's pipeline, textures and quads.
    fn prepare_bar(
        &mut self,
        display: &DisplayCore<P>,
        device: &P::Device,
        format: u64,
        size: (u64, u64),
        bar: &BarDraw,
    ) -> Option<PreparedBar<P>> {
        let Some(overlay) = display.overlay_pipeline(format) else {
            return self.refused("the overlay pipeline");
        };
        let platform = display.platform();
        if self.background.is_none() {
            self.background =
                rgba_texture(platform, device, self.own_texture, (1, 1), &BAR_BACKGROUND);
        }
        let Some(background) = self.background.as_ref().map(|t| t.clone_ref()) else {
            return self.refused("a texture for the bar");
        };
        let atlas = match &self.atlas {
            Some((drawn, texture)) if Arc::ptr_eq(drawn, &bar.atlas) => texture.clone_ref(),
            _ => {
                let mut rgba = bar.atlas.rgba.clone();
                super::cursor::premultiply(&mut rgba);
                let (width, height) = bar.atlas.size();
                let made = rgba_texture(
                    platform,
                    device,
                    self.own_texture,
                    (u64::from(width), u64::from(height)),
                    &rgba,
                );
                let Some(texture) = made else {
                    return self.refused("a texture for the bar's atlas");
                };
                self.atlas = Some((Arc::clone(&bar.atlas), texture.clone_ref()));
                texture
            }
        };
        let rect = clip(bar.layout.bar, size)?;
        let count = bar.atlas.count;
        let buttons = bar
            .layout
            .buttons
            .iter()
            .enumerate()
            .filter_map(|(i, &button)| {
                let lit = bar.lit == Some(i);
                cell_quad(button, i, lit, count, size)
            })
            .collect();
        Some(PreparedBar {
            overlay,
            background,
            rect,
            atlas,
            buttons,
        })
    }

    /// Logs, the first time, that the platform refused `what`: the frame is drawn plain.
    fn refused<T>(&mut self, what: &str) -> Option<T> {
        if !std::mem::replace(&mut self.refusal_logged, true) {
            log::warn!(
                target: "reims_vgpu::window",
                "the window draws no rotation or bar: the platform refused {what}"
            );
        }
        None
    }
}

impl<P: Platform> Prepared<P> {
    /// Where the display draws its frame: the turned frame's texture, all of it, where the image
    /// is turned; else `target` over `region`, as with no decoration.
    pub(crate) fn frame_target<'a>(
        &'a self,
        target: &'a P::Texture,
        region: PixelRect,
    ) -> (&'a P::Texture, PixelRect) {
        match &self.turned {
            Some((texture, _, _)) => (
                texture,
                PixelRect {
                    x: 0,
                    y: 0,
                    width: texture.width(),
                    height: texture.height(),
                },
            ),
            None => (target, region),
        }
    }

    /// Whether the window's pass clears the target first: where the display drew into the
    /// turned frame's texture, not into it.
    pub(crate) fn clears(&self) -> bool {
        self.turned.is_some()
    }

    /// The window's pass's quads, in order (the module doc), inside a target of `size` pixels.
    pub(crate) fn draws(&self, size: (u64, u64)) -> Vec<WindowDraw<'_, P>> {
        let mut draws = Vec::new();
        let quad = |pipeline, texture, viewport, scissor, rotation| {
            quad(
                &self.vertices,
                pipeline,
                texture,
                viewport,
                scissor,
                rotation,
            )
        };
        if let Some((texture, region, rotation)) = &self.turned
            && let Some(scissor) = clip(*region, size)
        {
            draws.push(quad(
                &self.copy,
                texture,
                viewport_of(*region),
                scissor,
                *rotation,
            ));
        }
        if let Some(bar) = &self.bar {
            let rect = PixelRect {
                x: bar.rect.x,
                y: bar.rect.y,
                width: bar.rect.width,
                height: bar.rect.height,
            };
            draws.push(quad(
                &self.copy,
                &bar.background,
                viewport_of(rect),
                bar.rect,
                Rotation::None,
            ));
            for &(viewport, scissor) in &bar.buttons {
                draws.push(quad(
                    &bar.overlay,
                    &bar.atlas,
                    viewport,
                    scissor,
                    Rotation::None,
                ));
            }
        }
        draws
    }
}

/// The quad drawing `texture` with `pipeline` over `viewport`, clipped by `scissor`, turned by
/// `rotation`, from the strip in `vertices`.
fn quad<'a, P: Platform>(
    vertices: &'a P::Buffer,
    pipeline: &'a P::RenderPipeline,
    texture: &'a P::Texture,
    viewport: MtlViewport,
    scissor: MtlScissorRect,
    rotation: Rotation,
) -> WindowDraw<'a, P> {
    WindowDraw {
        pipeline,
        texture,
        viewport,
        scissor,
        vertices,
        positions: POSITIONS,
        tex_coords: tex_coords_at(rotation),
    }
}

/// `rect` as a viewport.
fn viewport_of(rect: PixelRect) -> MtlViewport {
    #[expect(clippy::as_conversions, reason = "u64 to f64 rounds to nearest")]
    MtlViewport {
        origin_x: rect.x as f64,
        origin_y: rect.y as f64,
        width: rect.width as f64,
        height: rect.height as f64,
        znear: 0.0,
        zfar: 1.0,
    }
}

/// `rect` clipped to a target of `size` pixels, as a scissor: `None` where nothing is left.
fn clip(rect: PixelRect, size: (u64, u64)) -> Option<MtlScissorRect> {
    let x1 = rect.x.saturating_add(rect.width).min(size.0);
    let y1 = rect.y.saturating_add(rect.height).min(size.1);
    (x1 > rect.x && y1 > rect.y).then(|| MtlScissorRect {
        x: rect.x,
        y: rect.y,
        width: x1 - rect.x,
        height: y1 - rect.y,
    })
}

/// The quad that draws button `index` of `count`, lit or not, over `button` in a target of
/// `size` pixels (the module doc): the whole atlas scaled so a cell is the button, placed so the
/// button's cell (its row, the lit column or the normal one) lies over it, and the button as the
/// scissor. `None` where the button is off the target, and where the quad would reach past
/// [`VIEWPORT_BOUNDS`] or be larger than [`VIEWPORT_DIMENSION`].
fn cell_quad(
    button: PixelRect,
    index: usize,
    lit: bool,
    count: u32,
    size: (u64, u64),
) -> Option<(MtlViewport, MtlScissorRect)> {
    let scissor = clip(button, size)?;
    #[expect(clippy::as_conversions, reason = "pixel counts far below 2^52")]
    let (x, y, side, index) = (
        button.x as f64,
        button.y as f64,
        button.width as f64,
        index as f64,
    );
    let column = if lit { 1.0 } else { 0.0 };
    let viewport = MtlViewport {
        origin_x: x - column * side,
        origin_y: y - index * side,
        width: 2.0 * side,
        height: f64::from(count) * side,
        znear: 0.0,
        zfar: 1.0,
    };
    let (low, high) = VIEWPORT_BOUNDS;
    let within = |start: f64, length: f64| {
        start >= low && start + length <= high && length <= VIEWPORT_DIMENSION
    };
    (within(viewport.origin_x, viewport.width) && within(viewport.origin_y, viewport.height))
        .then_some((viewport, scissor))
}

/// A texture `own_texture` makes of `size` RGBA8 pixels `rgba`, read by shaders; `None` where the
/// platform makes none.
fn rgba_texture<P: Platform>(
    platform: &Arc<P>,
    device: &P::Device,
    own_texture: OwnTexture<P>,
    size: (u64, u64),
    rgba: &[u8],
) -> Option<P::Texture> {
    let (width, height) = size;
    let row = width.checked_mul(4)?;
    let length = usize::try_from(row.checked_mul(height)?).ok()?;
    if length == 0 || rgba.len() != length {
        return None;
    }
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
        HostRange::of_slice(rgba),
        row,
        row * height,
    ));
    Some(texture)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rectangle.
    fn rect(x: u64, y: u64, width: u64, height: u64) -> PixelRect {
        PixelRect {
            x,
            y,
            width,
            height,
        }
    }

    /// The vertex buffer holds the display's positions, then each rotation's texture
    /// coordinates, each block at a multiple of 256 bytes
    #[test]
    fn the_strip_s_blocks_lie_at_their_offsets() {
        let bytes = vertex_bytes();
        let floats = |at: u64, n: usize| -> Vec<f32> {
            let at = usize::try_from(at).unwrap();
            (0..n)
                .map(|i| f32::from_ne_bytes(bytes[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
                .collect()
        };
        assert_eq!(floats(POSITIONS, 16), POS_DATA);
        for r in [
            Rotation::None,
            Rotation::Quarter,
            Rotation::Half,
            Rotation::ThreeQuarters,
        ] {
            assert_eq!(tex_coords_at(r) % BLOCK, 0);
            assert_eq!(floats(tex_coords_at(r), 8), r.tex_coords());
        }
        assert_eq!(
            floats(tex_coords_at(Rotation::None), 8),
            reims_vgpu_core::display::TEX_DATA,
            "unturned, the display's own"
        );
    }

    /// A button's quad scales the atlas so a cell is the button, and puts the button's cell,
    /// its row and the lit or normal column, over it, clipped to the button
    #[test]
    fn a_button_s_quad_puts_its_cell_over_it() {
        let (v, s) = cell_quad(rect(408, 176, 48, 48), 1, false, 3, (464, 400)).unwrap();
        assert_eq!(
            (v.origin_x, v.origin_y, v.width, v.height),
            (408.0, 128.0, 96.0, 144.0)
        );
        assert_eq!((s.x, s.y, s.width, s.height), (408, 176, 48, 48));
        let (v, _) = cell_quad(rect(408, 176, 48, 48), 1, true, 3, (464, 400)).unwrap();
        assert_eq!((v.origin_x, v.origin_y), (360.0, 128.0));
        // Past the target's bottom: clipped, then not drawn at all.
        let (_, s) = cell_quad(rect(408, 380, 48, 48), 2, false, 3, (464, 400)).unwrap();
        assert_eq!(s.height, 20);
        assert!(cell_quad(rect(408, 400, 48, 48), 2, false, 3, (464, 400)).is_none());
        // A quad taller than Vulkan's least viewport is not drawn.
        assert!(cell_quad(rect(0, 0, 256, 256), 0, false, 17, (512, 8000)).is_none());
        assert!(cell_quad(rect(0, 0, 256, 256), 0, false, 16, (512, 8000)).is_some());
    }

    /// A rectangle clips to the target, and one off it is nothing
    #[test]
    fn a_rectangle_clips_to_the_target() {
        let s = clip(rect(10, 20, 100, 100), (50, 60)).unwrap();
        assert_eq!((s.x, s.y, s.width, s.height), (10, 20, 40, 40));
        assert!(clip(rect(50, 0, 10, 10), (50, 60)).is_none());
        assert!(clip(rect(0, 0, 0, 10), (50, 60)).is_none());
    }
}
