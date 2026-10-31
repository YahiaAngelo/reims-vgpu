//! What the window draws around the guest's image at its control's say (`control`, Linux's
//! `REIMS_VGPU_WINDOW_CONTROL`): the image turned a quarter, a half or three quarters clockwise
//! ([`Rotation`]), and a bar of buttons on the image's right ([`BarLayout`]). With no control,
//! or none of either asked for, the window is as it was without this module: nothing here is
//! drawn, laid out or hit.
//!
//! **The rotation** is the image's alone: the guest's mode, its frames and the pointer it is
//! sent are the guest's, unturned. The window lays the image out at its turned size
//! ([`Rotation::shown`]), the render thread draws the frame turned into that rectangle, and the
//! pointer's pixel under the host's pointer is turned back to the guest's
//! ([`Rotation::guest_pixel`]).
//!
//! **The bar** is a strip of the window's own right of the image: the window is the image's
//! width and the bar's ([`bar_width`]), and the image is laid out left of it, as inside an
//! inset. Its background is [`BAR_BACKGROUND`]; its buttons are square, [`Atlas::cell`] points a
//! side, stacked [`BUTTON_SPACING`] apart and centred down the bar, [`BAR_PADDING`] from its
//! edges. Each button's image is a cell of the control's atlas ([`Atlas`]), its normal one, or
//! its lit one while the pointer is over it or pressing it. The bar is laid out in the drawable's
//! pixels, its points times the window's pixels per point, so it is the same size on a screen of
//! any scale ([`BarLayout::new`]), and the pointer is hit-tested on those same pixels
//! ([`BarLayout::button_at`]). What the render thread draws is a [`Decoration`].

use std::sync::Arc;

use crate::handle::PixelRect;

/// The bar's background, RGBA: a dark grey (#1c1c1e).
pub const BAR_BACKGROUND: [u8; 4] = [0x1c, 0x1c, 0x1e, 0xff];

/// Points between a button and the bar's left and right edges.
pub const BAR_PADDING: f64 = 8.0;

/// Points between two buttons.
pub const BUTTON_SPACING: f64 = 8.0;

/// How the guest's image is turned, clockwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// As the guest draws it.
    #[default]
    None,
    /// A quarter turn: the guest's top is on the right.
    Quarter,
    /// A half turn: upside down.
    Half,
    /// Three quarters: the guest's top is on the left.
    ThreeQuarters,
}

impl Rotation {
    /// The rotation of `degrees` clockwise: 0, 90, 180 or 270, else none.
    pub fn from_degrees(degrees: u32) -> Option<Rotation> {
        match degrees {
            0 => Some(Rotation::None),
            90 => Some(Rotation::Quarter),
            180 => Some(Rotation::Half),
            270 => Some(Rotation::ThreeQuarters),
            _ => None,
        }
    }

    /// Its degrees clockwise.
    pub fn degrees(self) -> u32 {
        match self {
            Rotation::None => 0,
            Rotation::Quarter => 90,
            Rotation::Half => 180,
            Rotation::ThreeQuarters => 270,
        }
    }

    /// Whether it swaps the image's width and height.
    pub fn swaps(self) -> bool {
        matches!(self, Rotation::Quarter | Rotation::ThreeQuarters)
    }

    /// A guest `size` (width, height) as the window shows it: swapped by a quarter turn either
    /// way.
    pub fn shown<T>(self, size: (T, T)) -> (T, T) {
        if self.swaps() { (size.1, size.0) } else { size }
    }

    /// The guest pixel shown at pixel `shown` (column, row) of the turned image of a guest mode
    /// `mode` (width, height): `shown` is inside [`Rotation::shown`] of `mode`, as
    /// [`crate::geometry::pointer_to_guest`] gives it for the turned mode, and the answer inside
    /// `mode`. A quarter turn shows the guest's pixel (x, y) at column `height - 1 - y`, row `x`.
    pub fn guest_pixel(self, shown: (u16, u16), mode: (u16, u16)) -> (u16, u16) {
        let (a, b) = shown;
        let last = |size: u16, at: u16| size.saturating_sub(1).saturating_sub(at);
        match self {
            Rotation::None => (a, b),
            Rotation::Quarter => (b, last(mode.1, a)),
            Rotation::Half => (last(mode.0, a), last(mode.1, b)),
            Rotation::ThreeQuarters => (last(mode.0, b), a),
        }
    }

    /// The texture coordinates of the strip's four vertices (bottom left, top left, bottom
    /// right, top right of the turned image on screen; the texture's y down) that show the
    /// guest's image turned: the display's own (`TEX_DATA`) for none.
    pub fn tex_coords(self) -> [f32; 8] {
        match self {
            Rotation::None => [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            Rotation::Quarter => [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            Rotation::Half => [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            Rotation::ThreeQuarters => [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    }
}

/// The bar's buttons' images: RGBA8, not premultiplied, rows top to bottom with no padding,
/// `2 * cell` pixels wide and `count * cell` high. Row of cells `i` is button `i`: its normal
/// image on the left, its lit one (hovered or pressed) on the right.
#[derive(Clone, PartialEq, Eq)]
pub struct Atlas {
    /// A button's side, in pixels of the atlas and in points of the window.
    pub cell: u32,
    /// The buttons.
    pub count: u32,
    /// The pixels.
    pub rgba: Vec<u8>,
}

impl std::fmt::Debug for Atlas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Atlas")
            .field("cell", &self.cell)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

/// The most buttons a bar holds, and the largest button side, in points: a bar of more or
/// larger would not fit a screen, and the render thread draws each button's cell from a quad of
/// the whole atlas, which must stay within Vulkan's least viewport size.
pub const MAX_BUTTONS: u32 = 32;
/// The largest button side, in points.
pub const MAX_CELL: u32 = 256;

impl Atlas {
    /// The atlas of `count` buttons of `cell` pixels a side in `rgba`, or why it is none: no
    /// button, a cell of no pixels or past [`MAX_CELL`], more than [`MAX_BUTTONS`], or pixels
    /// that are not exactly `2 * cell` by `count * cell`.
    pub fn new(cell: u32, count: u32, rgba: Vec<u8>) -> Result<Atlas, String> {
        if count == 0 || count > MAX_BUTTONS {
            return Err(format!("{count} buttons: 1 to {MAX_BUTTONS} are drawn"));
        }
        if cell == 0 || cell > MAX_CELL {
            return Err(format!(
                "a button of {cell} pixels: 1 to {MAX_CELL} are drawn"
            ));
        }
        let expected = u64::from(cell) * 2 * u64::from(cell) * u64::from(count) * 4;
        if u64::try_from(rgba.len()).ok() != Some(expected) {
            return Err(format!(
                "the atlas is {} bytes, not {expected} ({} by {} RGBA8 pixels)",
                rgba.len(),
                2 * cell,
                count * cell
            ));
        }
        Ok(Atlas { cell, count, rgba })
    }

    /// Its width and height in pixels.
    pub fn size(&self) -> (u32, u32) {
        (2 * self.cell, self.count * self.cell)
    }
}

/// The bar's width in points for buttons of `cell` points a side: the button and its padding.
pub fn bar_width(cell: u32) -> f64 {
    f64::from(cell) + 2.0 * BAR_PADDING
}

/// The bar's width in whole pixels for buttons of `cell` points a side at `scale` pixels a
/// point: what [`BarLayout::new`] lays out, and what the window keeps clear for it.
pub fn bar_pixels(cell: u32, scale: f64) -> u64 {
    pixels(bar_width(cell), scale)
}

/// `points` at `scale` pixels a point, in whole pixels.
fn pixels(points: f64, scale: f64) -> u64 {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let p = points * scale;
    if p.is_finite() && p > 0.0 {
        p.round() as u64
    } else {
        0
    }
}

/// Where the bar and its buttons are in a drawable, in its pixels (the module doc).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BarLayout {
    /// The bar: the drawable's right edge, all of its height.
    pub bar: PixelRect,
    /// Each button, in the atlas's order, stacked down the bar and centred in it; from the top
    /// where they are more than the drawable's height holds, the last past its bottom.
    pub buttons: Vec<PixelRect>,
}

impl BarLayout {
    /// The bar for `count` buttons of `cell` points in a drawable of `drawable` pixels at
    /// `scale` pixels a point (the module doc). A drawable narrower than the bar holds the bar
    /// alone.
    pub fn new(drawable: (u64, u64), scale: f64, cell: u32, count: u32) -> BarLayout {
        let width = bar_pixels(cell, scale).min(drawable.0);
        let bar = PixelRect {
            x: drawable.0 - width,
            y: 0,
            width,
            height: drawable.1,
        };
        let side = pixels(f64::from(cell), scale);
        let spacing = pixels(BUTTON_SPACING, scale);
        let count = u64::from(count);
        let total = side * count + spacing * count.saturating_sub(1);
        let top = drawable.1.saturating_sub(total) / 2;
        let x = bar.x + width.saturating_sub(side) / 2;
        let buttons = (0..count)
            .map(|i| PixelRect {
                x,
                y: top + i * (side + spacing),
                width: side,
                height: side,
            })
            .collect();
        BarLayout { bar, buttons }
    }

    /// Whether `point`, in the drawable's pixels, is on the bar.
    pub fn on_bar(&self, point: (f64, f64)) -> bool {
        inside(self.bar, point)
    }

    /// The button under `point`, in the drawable's pixels.
    pub fn button_at(&self, point: (f64, f64)) -> Option<usize> {
        self.buttons.iter().position(|&b| inside(b, point))
    }
}

/// Whether `point` is in `rect`: its left and top edges in, its right and bottom out.
fn inside(rect: PixelRect, point: (f64, f64)) -> bool {
    let (x, y) = (rect.x as f64, rect.y as f64);
    point.0 >= x
        && point.0 < x + rect.width as f64
        && point.1 >= y
        && point.1 < y + rect.height as f64
}

/// The bar as the render thread draws it. Two are equal where they lay out and light the same,
/// over the same atlas (the same `Arc`, not compared pixel by pixel).
#[derive(Clone, Debug)]
pub struct BarDraw {
    /// Where the bar and its buttons are.
    pub layout: BarLayout,
    /// The buttons' images.
    pub atlas: Arc<Atlas>,
    /// The button shown lit, where one is.
    pub lit: Option<usize>,
}

impl PartialEq for BarDraw {
    fn eq(&self, other: &BarDraw) -> bool {
        self.layout == other.layout
            && self.lit == other.lit
            && Arc::ptr_eq(&self.atlas, &other.atlas)
    }
}

/// What the render thread draws around the guest's frame (the module doc): the frame turned
/// by `rotation` into the image's rectangle, and the bar where there is one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Decoration {
    /// How the frame is turned.
    pub rotation: Rotation,
    /// The bar, where it shows.
    pub bar: Option<BarDraw>,
}

impl Decoration {
    /// Whether it draws anything of its own: a turn or a bar. A window with none draws as one
    /// with no control.
    pub fn is_plain(&self) -> bool {
        self.rotation == Rotation::None && self.bar.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{self, Insets};

    const ALL: [Rotation; 4] = [
        Rotation::None,
        Rotation::Quarter,
        Rotation::Half,
        Rotation::ThreeQuarters,
    ];

    /// The rotations are 0, 90, 180 and 270 degrees clockwise, and no other; a quarter turn
    /// either way swaps the image's sides
    #[test]
    fn the_rotations_are_the_four_quarter_turns() {
        for r in ALL {
            assert_eq!(Rotation::from_degrees(r.degrees()), Some(r));
        }
        for bad in [1, 45, 360, 450, u32::MAX] {
            assert_eq!(Rotation::from_degrees(bad), None);
        }
        assert_eq!(Rotation::Quarter.shown((1170u16, 2532u16)), (2532, 1170));
        assert_eq!(Rotation::Half.shown((1170u16, 2532u16)), (1170, 2532));
        assert_eq!(Rotation::ThreeQuarters.shown((3.0, 4.0)), (4.0, 3.0));
    }

    /// Where a guest pixel shows on the turned image, turned clockwise by `rotation`: its
    /// column and row there (the inverse of [`Rotation::guest_pixel`], written from the turn).
    fn shown_at(rotation: Rotation, (x, y): (u16, u16), (w, h): (u16, u16)) -> (u16, u16) {
        match rotation {
            Rotation::None => (x, y),
            Rotation::Quarter => (h - 1 - y, x),
            Rotation::Half => (w - 1 - x, h - 1 - y),
            Rotation::ThreeQuarters => (y, w - 1 - x),
        }
    }

    /// Each turn's pixel under the pointer turns back to the guest's: the guest's corners show
    /// at the corners a clockwise turn puts them at, and every pixel of a small mode goes and
    /// comes back
    #[test]
    fn a_turned_pixel_turns_back_to_the_guest_s() {
        let mode = (4, 3);
        // The guest's top left shows top right after a quarter turn, bottom right after a half,
        // bottom left after three quarters.
        assert_eq!(Rotation::Quarter.guest_pixel((2, 0), mode), (0, 0));
        assert_eq!(Rotation::Half.guest_pixel((3, 2), mode), (0, 0));
        assert_eq!(Rotation::ThreeQuarters.guest_pixel((0, 3), mode), (0, 0));
        // Its top right shows bottom right after a quarter turn.
        assert_eq!(Rotation::Quarter.guest_pixel((2, 3), mode), (3, 0));
        for r in ALL {
            for x in 0..mode.0 {
                for y in 0..mode.1 {
                    let shown = shown_at(r, (x, y), mode);
                    let (sw, sh) = r.shown(mode);
                    assert!(shown.0 < sw && shown.1 < sh, "{r:?} {shown:?}");
                    assert_eq!(r.guest_pixel(shown, mode), (x, y), "{r:?} ({x}, {y})");
                }
            }
        }
    }

    /// The texture coordinates put the guest's corners where the turn puts them: the vertex at
    /// a screen corner samples the guest corner that [`shown_at`] puts there
    #[test]
    fn the_texture_coordinates_turn_the_image() {
        // The strip's vertices: bottom left, top left, bottom right, top right, as fractions of
        // the shown image (x right, y down).
        let corners = [(0.0, 1.0), (0.0, 0.0), (1.0, 1.0), (1.0, 0.0)];
        let mode = (2, 2);
        for r in ALL {
            let tex = r.tex_coords();
            for (i, (sx, sy)) in corners.into_iter().enumerate() {
                let (u, v) = (tex[2 * i], tex[2 * i + 1]);
                // The guest corner at (u, v), as a pixel of a 2 by 2 mode, shows at the screen
                // corner (sx, sy).
                let guest = (u as u16, v as u16);
                let shown = shown_at(r, guest, mode);
                assert_eq!(
                    (f64::from(shown.0), f64::from(shown.1)),
                    (sx, sy),
                    "{r:?} vertex {i}"
                );
            }
        }
        assert_eq!(
            Rotation::None.tex_coords(),
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            "the display's own"
        );
    }

    /// The pointer over a turned image, laid out at its turned size, reaches the guest pixel
    /// drawn under it, with and without the bar beside it, at 1× and 2×; the bar is not the
    /// image
    #[test]
    fn the_pointer_over_a_turned_image_reaches_the_pixel_under_it() {
        let mode = (100, 200);
        for scale in [1.0, 2.0] {
            for bar in [0.0, bar_width(48)] {
                for r in ALL {
                    let (sw, sh) = r.shown(mode);
                    let bounds = (f64::from(sw) + bar, f64::from(sh));
                    let insets = Insets {
                        right: bar,
                        ..Insets::default()
                    };
                    let l = geometry::layout(bounds, scale, insets, Some((sw, sh)));
                    assert_eq!(
                        (l.image.width, l.image.height),
                        (f64::from(sw), f64::from(sh)),
                        "{r:?} at {scale}×: a shown point per guest pixel"
                    );
                    // The guest's pixel (10, 20), at its centre on screen.
                    let shown = shown_at(r, (10, 20), mode);
                    let point = (
                        l.image.x + f64::from(shown.0) + 0.5,
                        l.image.y + f64::from(shown.1) + 0.5,
                    );
                    let pixel = geometry::pointer_to_guest(point, l.image, (sw, sh));
                    assert_eq!(r.guest_pixel(pixel, mode), (10, 20), "{r:?} {scale} {bar}");
                    // The image's corners are its first and last pixels, turned back.
                    let corner = (l.image.x, l.image.y);
                    let pixel = geometry::pointer_to_guest(corner, l.image, (sw, sh));
                    let top_left = r.guest_pixel(pixel, mode);
                    assert_eq!(shown_at(r, top_left, mode), (0, 0));
                    if bar > 0.0 {
                        let on_bar = (bounds.0 - 1.0, bounds.1 / 2.0);
                        assert!(!geometry::contains(l.image, on_bar));
                    }
                }
            }
        }
    }

    /// The atlas is two cells across and a cell a button down, RGBA8, exactly; no button, an
    /// empty or too large cell and too many buttons are refused
    #[test]
    fn the_atlas_is_two_cells_by_one_a_button() {
        let ok = Atlas::new(4, 3, vec![0; 8 * 12 * 4]).unwrap();
        assert_eq!(ok.size(), (8, 12));
        assert!(Atlas::new(4, 3, vec![0; 8 * 12 * 4 - 1]).is_err());
        assert!(Atlas::new(4, 3, vec![0; 8 * 12 * 4 + 4]).is_err());
        assert!(Atlas::new(4, 0, Vec::new()).is_err());
        assert!(Atlas::new(0, 3, Vec::new()).is_err());
        assert!(Atlas::new(MAX_CELL + 1, 1, vec![0; 2 * 257 * 257 * 4]).is_err());
        let many = MAX_BUTTONS + 1;
        assert!(Atlas::new(1, many, vec![0; 2 * many as usize * 4]).is_err());
    }

    /// The bar is the button and its padding wide at the drawable's right, its buttons centred
    /// down it and across it, a spacing apart, at the window's scale
    #[test]
    fn the_bar_s_buttons_are_stacked_and_centred() {
        assert_eq!(bar_width(48), 64.0);
        let l = BarLayout::new((464, 400), 1.0, 48, 3);
        assert_eq!(
            l.bar,
            PixelRect {
                x: 400,
                y: 0,
                width: 64,
                height: 400
            }
        );
        // 3 × 48 + 2 × 8 = 160 high, from 120 to 280.
        let ys: Vec<u64> = l.buttons.iter().map(|b| b.y).collect();
        assert_eq!(ys, [120, 176, 232]);
        assert!(l.buttons.iter().all(|b| b.x == 408 && b.width == 48));
        // At 2×, every length doubles.
        let l2 = BarLayout::new((928, 800), 2.0, 48, 3);
        assert_eq!((l2.bar.x, l2.bar.width), (800, 128));
        let ys: Vec<u64> = l2.buttons.iter().map(|b| b.y).collect();
        assert_eq!(ys, [240, 352, 464]);
        assert!(l2.buttons.iter().all(|b| b.x == 816 && b.width == 96));
        // Taller than the drawable: from its top, the last past its bottom.
        let tall = BarLayout::new((64, 100), 1.0, 48, 3);
        let ys: Vec<u64> = tall.buttons.iter().map(|b| b.y).collect();
        assert_eq!(ys, [0, 56, 112]);
    }

    /// A point hits the button under it, its left and top edges in, its right and bottom out;
    /// the padding and the spacing hit none but are on the bar, and the image is not the bar
    #[test]
    fn a_point_hits_the_button_under_it() {
        let l = BarLayout::new((464, 400), 1.0, 48, 3);
        assert_eq!(l.button_at((408.0, 120.0)), Some(0));
        assert_eq!(l.button_at((455.9, 167.9)), Some(0));
        assert_eq!(l.button_at((456.0, 150.0)), None);
        assert_eq!(l.button_at((430.0, 170.0)), None, "the spacing");
        assert_eq!(l.button_at((430.0, 176.0)), Some(1));
        assert_eq!(l.button_at((430.0, 279.9)), Some(2));
        assert_eq!(l.button_at((430.0, 280.0)), None);
        assert_eq!(l.button_at((404.0, 200.0)), None, "the padding");
        assert!(l.on_bar((404.0, 200.0)) && l.on_bar((400.0, 0.0)));
        assert!(!l.on_bar((399.9, 200.0)));
        assert_eq!(l.button_at((200.0, 200.0)), None);
        // At 2× the point is in the drawable's pixels.
        let l2 = BarLayout::new((928, 800), 2.0, 48, 3);
        assert_eq!(l2.button_at((2.0 * 430.0, 2.0 * 176.0)), Some(1));
    }

    /// No turn and no bar is a plain window
    #[test]
    fn no_turn_and_no_bar_is_plain() {
        assert!(Decoration::default().is_plain());
        let turned = Decoration {
            rotation: Rotation::Half,
            bar: None,
        };
        assert!(!turned.is_plain());
    }
}
