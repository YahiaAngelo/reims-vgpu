//! The modes, the size in millimetres and the scale factor a display matching its host screen
//! offers the guest (`WindowConfig::match_host_screen`), and the scale a guest runs a mode at.
//!
//! **A chosen mode** (`WindowConfig::mode`, QEMU's `resolution`, 1920×1080 by default there):
//! the mode first, then the standard sizes of each ratio within it ([`modes_up_to`]), at 110
//! ppi, a 1× guest desktop ([`density_1x`]). The window scales it to its size on the GPU. A
//! density chosen with it (`WindowConfig::mode_ppi`, QEMU's `resolution=2560x1600@200`) is told
//! in place of 110 ppi ([`density_chosen`]): where the guest reads it as 146.67 ppi or more
//! ([`guest_scale`]), it runs a HiDPI desktop at the mode's pixels.
//!
//! **The screen's own modes** (no mode chosen) fill the screen: the largest 16:10, 16:9 and 4:3
//! rectangles within its pixels, 16:10 first (the one the guest prefers), then the standard
//! smaller sizes of each ratio ([`offered_modes`]). Their density follows the screen's backing
//! scale: 110 pixels per inch per unit of scale, so a Retina screen (scale 2) is told 220 ppi
//! and gets a HiDPI guest desktop, and a 1× screen is told 110 ppi and gets a 1× one
//! ([`density`]).
//!
//! Either way the modes run at the screen's refresh rate (its current mode's, the rate the
//! person set it to, `crate::screens`), or at `REIMS_VGPU_WINDOW_REFRESH`'s, and the size in millimetres is the first mode's pixels at the
//! density. The scale factor is 2 above 143 ppi for the screen's own modes, and for a chosen
//! density the scale the guest runs the mode at ([`guest_scale`]).
//!
//! **Without a screen** (as for a VM started from an ssh shell, with no window system to reach):
//! the chosen mode as above, or [`UNMATCHED_MODE`] at 1× where none is chosen, at 60 Hz
//! ([`DEFAULT_REFRESH_HZ`]) or `REIMS_VGPU_WINDOW_REFRESH`'s. A display that offered nothing
//! would leave the guest no mode to set, and macOS stops its boot at the Apple logo.
//!
//! **The guest's scale.** Measured with a macOS 13.6 guest (22G120) on the M3 Max's built-in
//! screen, each boot from a fresh clone, so that no display settings were saved for it (the
//! scale shows in the menu bar's height in the guest's pixels: 24 at 1×, 48 at HiDPI):
//! - the guest reads the width it is told in whole centimetres: rounded to whole millimetres,
//!   as the display's settings page holds it, then up to whole centimetres. 611.04 mm reads as
//!   62 cm, 590.54 as 60 (591 mm), 446.42 as 45 and 440.39 as 44 (440 mm);
//! - it picks its default from the pixels per inch across that width, pixels / (centimetres ×
//!   10 / 25.4): 1× (the mode at 1×) at up to 3811 pixels across 66 cm (146.666 ppi), HiDPI
//!   (looking like half the mode) from 2021 pixels across 35 cm (146.667 ppi), for modes from
//!   1920 to 3811 pixels wide:
//!   - 1× at 141.6 and 143.9 (3456 pixels across 62 and 61 cm), 144.5 (2560 across 45), 146.30
//!     (3456 across 60), 146.62 (2944 across 51), 146.657 (2656 across 46), 146.665 (1790
//!     across 31) and 146.666 (3811 across 66);
//!   - HiDPI at 146.667 (2021 across 35), 146.673 (3176 across 55), 146.76 (2080 across 36),
//!     146.93 (3008 across 52), 147.8 (2560 across 44, 1920 across 33), 148.1 (2624 across 45),
//!     148.6 (2048 across 35), 216.7 (5120 across 60), 224.2 (2560 across 29), and 148.8,
//!     159.6, 219.5 and 250.8 (3456 across 59, 55, 40 and 35);
//!   - between 146.666 and 146.667 it was not measured, and [`guest_scale`] takes it as 1×;
//! - the height does not change it: 3456 across 60 cm is 1× with 2160 pixels across 37 cm
//!   (148.3 ppi), and 2048 across 35 is HiDPI with 1536 across 27 (144.5);
//! - neither does the page's scale factor: told 254 ppi with a scale factor of 1, it still runs
//!   at 2×. The guest never sent its compositor's parameters;
//! - a much wider mode does not follow it: 5832×3645 ran HiDPI where it read 128.8 ppi (115 cm)
//!   as well as at 146.666 (101 cm), and ran 1× nowhere. Where between 3811 and 5832 pixels
//!   the guest turns HiDPI below 146.667 ppi, 110 ppi included, was not measured, and
//!   [`guest_scale`] takes such a mode as 1×;
//! - nor does a narrow mode: 1280×800 ran 1× where it read 216.7 ppi across 15 cm (HiDPI would
//!   look like 640×400). HiDPI was measured from 1920 pixels wide (33 cm) and from 29 cm (2560
//!   pixels); whether the pixels or the centimetres kept 1280×800 at 1×, and where between,
//!   was not measured. [`guest_scale`] takes a mode narrower than 1920 pixels as 1×, and a
//!   size below 29 cm by its density;
//! - so 72 ppi per unit of scale, 144 for a Retina screen, read as 141.6, gave that screen a 1×
//!   desktop; 110 per unit of scale puts 1× and 2× screens each well on their own side.

use crate::screens::HostScreen;

/// A mode offered to the guest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OfferedMode {
    /// Pixels wide.
    pub width: u16,
    /// Pixels high.
    pub height: u16,
    /// The refresh rate, in Hz.
    pub refresh_hz: f64,
}

/// What the guest is told of the display's density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Density {
    /// Width, then height.
    pub size_in_millimeters: (f64, f64),
    /// The descriptor's scale factor: 2 for a HiDPI screen, else 1. A macOS 13.6 guest picks its
    /// default from the size alone (module doc).
    pub scale_factor: u8,
}

/// A ratio the guest is offered, and its standard sizes.
struct Ratio {
    /// Width to height.
    ratio: (u64, u64),
    /// Its standard sizes, largest first.
    sizes: &'static [(u16, u16)],
}

/// The ratios offered, in the order their largest modes are listed.
const RATIOS: [Ratio; 3] = [
    Ratio {
        ratio: (16, 10),
        sizes: &[
            (2560, 1600),
            (1920, 1200),
            (1680, 1050),
            (1440, 900),
            (1280, 800),
        ],
    },
    Ratio {
        ratio: (16, 9),
        sizes: &[
            (3840, 2160),
            (2560, 1440),
            (1920, 1080),
            (1600, 900),
            (1280, 720),
        ],
    },
    Ratio {
        ratio: (4, 3),
        sizes: &[(2048, 1536), (1600, 1200), (1400, 1050), (1024, 768)],
    },
];

/// The most modes offered.
pub const MAX_MODES: usize = 128;

/// The rate used where neither the override nor the screen gives one, in Hz.
pub const DEFAULT_REFRESH_HZ: f64 = 60.0;

/// The mode offered first, at 1×, where there is no screen to match and no mode is chosen:
/// QEMU's default `resolution`.
pub const UNMATCHED_MODE: (u16, u16) = (1920, 1080);

/// The pixels per inch the guest is told per unit of the screen's backing scale.
const PPI_PER_SCALE: f64 = 110.0;

/// Above this many pixels per inch the scale factor told for the screen's own modes is 2.
const HIDPI_ABOVE_PPI: f64 = 143.0;

/// The least density a macOS 13.6 guest was measured to run a mode 1920 to 3811 pixels wide
/// HiDPI at, as pixels across whole centimetres: 2021 across 35 (146.667 ppi). It ran 1× at
/// 3811 across 66 (146.666 ppi), the most measured there (module doc); a wider mode ran HiDPI
/// below both.
const HIDPI_FROM: (f64, f64) = (2021.0, 35.0);

/// The narrowest mode a macOS 13.6 guest was measured to run HiDPI, in pixels: 1920 across 33
/// cm. 1280 across 15 cm ran 1× at 216.7 ppi (module doc).
const HIDPI_NARROWEST: u16 = 1920;

/// The divisor of pixels per inch that gives pixels per millimetre. It is 25.46, not the 25.4
/// millimetres of an inch, and kept so: changing it changes the size every guest is told.
const MILLIMETERS_DIVISOR: f64 = 25.46;

/// The largest `ratio` rectangle within `pixels`, each side rounded down to a multiple of 8.
/// `None` where that leaves a side empty.
fn largest(pixels: (u64, u64), ratio: (u64, u64)) -> Option<(u16, u16)> {
    let (pw, ph) = pixels;
    let (rw, rh) = ratio;
    let (w, h) = if pw * rh <= ph * rw {
        (pw, pw * rh / rw)
    } else {
        (ph * rw / rh, ph)
    };
    let (w, h) = (w & !7, h & !7);
    if w == 0 || h == 0 {
        return None;
    }
    Some((u16::try_from(w).ok()?, u16::try_from(h).ok()?))
}

/// The rate the modes run at: `refresh_hz` where given, else the screen's, else
/// [`DEFAULT_REFRESH_HZ`].
fn refresh(screen: Option<&HostScreen>, refresh_hz: Option<f64>) -> f64 {
    let valid = |hz: f64| hz.is_finite() && hz > 0.0;
    match (refresh_hz, screen) {
        (Some(hz), _) if valid(hz) => hz,
        (_, Some(screen)) if valid(screen.refresh_hz) => screen.refresh_hz,
        _ => DEFAULT_REFRESH_HZ,
    }
}

/// The modes offered for `screen`, the first the preferred one. `refresh_hz` overrides the
/// screen's rate. Empty for a screen with no pixels.
///
/// First the largest 16:10, 16:9 and 4:3 rectangles within the screen's pixels, each side
/// rounded down to a multiple of 8; then, for each ratio in that order, its standard sizes
/// smaller than its largest. No mode is listed twice; every side is below 65536; there are at
/// most [`MAX_MODES`].
pub fn offered_modes(screen: &HostScreen, refresh_hz: Option<f64>) -> Vec<OfferedMode> {
    let refresh_hz = refresh(Some(screen), refresh_hz);
    let limit = u64::from(u16::MAX);
    let pixels = (
        u64::from(screen.pixels.0).min(limit),
        u64::from(screen.pixels.1).min(limit),
    );
    let largest: Vec<Option<(u16, u16)>> =
        RATIOS.iter().map(|r| largest(pixels, r.ratio)).collect();
    let smaller = RATIOS.iter().zip(&largest).flat_map(|(r, top)| {
        r.sizes
            .iter()
            .copied()
            .filter(move |&(w, h)| top.is_some_and(|(top_w, top_h)| w < top_w && h < top_h))
    });
    let mut modes: Vec<OfferedMode> = Vec::new();
    for (width, height) in largest.iter().flatten().copied().chain(smaller) {
        if modes.len() == MAX_MODES {
            break;
        }
        if modes.iter().any(|m| (m.width, m.height) == (width, height)) {
            continue;
        }
        modes.push(OfferedMode {
            width,
            height,
            refresh_hz,
        });
    }
    modes
}

/// The modes offered for a chosen mode `mode` (width, height) on `screen`: `mode` first, then,
/// for each ratio in order, its standard sizes within `mode` (neither side larger). No mode is
/// listed twice. `refresh_hz` overrides the screen's rate, as for [`offered_modes`].
/// Empty for a mode with an empty side. With no `screen`, the rate is `refresh_hz`, or else
/// [`DEFAULT_REFRESH_HZ`].
pub fn modes_up_to(
    mode: (u16, u16),
    screen: Option<&HostScreen>,
    refresh_hz: Option<f64>,
) -> Vec<OfferedMode> {
    if mode.0 == 0 || mode.1 == 0 {
        return Vec::new();
    }
    let refresh_hz = refresh(screen, refresh_hz);
    let within = RATIOS
        .iter()
        .flat_map(|r| r.sizes.iter().copied())
        .filter(|&(w, h)| w <= mode.0 && h <= mode.1);
    let mut modes: Vec<OfferedMode> = Vec::new();
    for (width, height) in std::iter::once(mode).chain(within) {
        if modes.iter().any(|m| (m.width, m.height) == (width, height)) {
            continue;
        }
        modes.push(OfferedMode {
            width,
            height,
            refresh_hz,
        });
    }
    modes
}

/// The density of `screen` for a guest mode `mode` (width, height) in pixels: ppi is 110 times
/// the screen's backing scale; the size is the mode's pixels at that density (divided by
/// `ppi / 25.46`); the scale factor is 2 above 143 ppi, else 1.
pub fn density(screen: &HostScreen, mode: (u16, u16)) -> Density {
    let scale = if screen.scale.is_finite() && screen.scale > 0.0 {
        screen.scale
    } else {
        1.0
    };
    density_at(PPI_PER_SCALE * scale, mode)
}

/// The density of a chosen mode `mode`: 110 ppi, whatever the screen, so the guest runs a 1×
/// desktop, which the window scales on the GPU.
pub fn density_1x(mode: (u16, u16)) -> Density {
    density_at(PPI_PER_SCALE, mode)
}

/// The density of a chosen mode `mode` told `ppi` pixels per inch: the size is the mode's
/// pixels at `ppi`, and the scale factor the scale the guest runs the mode at on that size
/// ([`guest_scale`]), which the guest does not follow but the window does. `None`, or a `ppi`
/// that is not a positive finite number, is 110 ppi ([`density_1x`]).
pub fn density_chosen(mode: (u16, u16), ppi: Option<f64>) -> Density {
    match ppi {
        Some(ppi) if ppi.is_finite() && ppi > 0.0 => {
            let at = density_at(ppi, mode);
            Density {
                scale_factor: guest_scale(mode, at.size_in_millimeters),
                ..at
            }
        }
        _ => density_1x(mode),
    }
}

/// A mode `mode` at `ppi` pixels per inch.
fn density_at(ppi: f64, mode: (u16, u16)) -> Density {
    let per_millimeter = ppi / MILLIMETERS_DIVISOR;
    Density {
        size_in_millimeters: (
            f64::from(mode.0) / per_millimeter,
            f64::from(mode.1) / per_millimeter,
        ),
        scale_factor: if ppi > HIDPI_ABOVE_PPI { 2 } else { 1 },
    }
}

/// The whole centimetres a guest reads a told width of `millimeters` as: rounded to whole
/// millimetres as the display's settings page holds it (its `float`, plus 0.5, truncated), then
/// up to whole centimetres. `None` for a width that is not a positive finite number, or that
/// comes to no centimetre.
fn guest_centimeters(millimeters: f64) -> Option<f64> {
    if !(millimeters.is_finite() && millimeters > 0.0) {
        return None;
    }
    let whole = (f64::from(millimeters as f32) + 0.5).floor();
    let centimeters = (whole / 10.0).ceil();
    (centimeters.is_finite() && centimeters > 0.0).then_some(centimeters)
}

/// The scale a guest runs a mode `mode` at on a display told `size_in_millimeters`, as a macOS
/// 13.6 guest picks it (module doc): it reads the width in whole centimetres, rounded to whole
/// millimetres and then up, and runs HiDPI where the mode's width comes to 2021 pixels per 35
/// cm (146.667 ppi) or more across it, for a mode at least 1920 pixels wide. 2 there, else 1,
/// and 1 for an empty or unknown size. The unmeasured densities just below, modes wider than
/// 3811 pixels below it, where the guest may run HiDPI anyway, and modes narrower than 1920
/// pixels, which 1280 was not, are 1. The window shows a guest point per host point by it.
pub fn guest_scale(mode: (u16, u16), size_in_millimeters: (f64, f64)) -> u8 {
    if mode.0 < HIDPI_NARROWEST {
        return 1;
    }
    let Some(centimeters) = guest_centimeters(size_in_millimeters.0) else {
        return 1;
    };
    // Pixels per centimetre at least 2021 / 35, cross-multiplied: whole numbers, exact in f64
    // wherever the two sides are close.
    let (pixels, across) = HIDPI_FROM;
    if f64::from(mode.0) * across >= pixels * centimeters {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen(pixels: (u32, u32), scale: f64, refresh_hz: f64) -> HostScreen {
        HostScreen {
            id: 1,
            pixels,
            scale,
            refresh_hz,
        }
    }

    fn sizes(modes: &[OfferedMode]) -> Vec<(u16, u16)> {
        modes.iter().map(|m| (m.width, m.height)).collect()
    }

    /// A 3456×2234 built-in Retina screen at 120 Hz
    #[test]
    fn offered_modes_for_a_built_in_retina_screen() {
        let modes = offered_modes(&screen((3456, 2234), 2.0, 120.0), None);
        assert_eq!(
            sizes(&modes),
            vec![
                (3456, 2160),
                (3456, 1944),
                (2976, 2232),
                (2560, 1600),
                (1920, 1200),
                (1680, 1050),
                (1440, 900),
                (1280, 800),
                (2560, 1440),
                (1920, 1080),
                (1600, 900),
                (1280, 720),
                (2048, 1536),
                (1600, 1200),
                (1400, 1050),
                (1024, 768),
            ]
        );
        assert!(modes.iter().all(|m| m.refresh_hz == 120.0));
    }

    /// A 5120×2880 screen: its largest 16:9 is the whole screen
    #[test]
    fn offered_modes_for_a_5k_screen() {
        let modes = offered_modes(&screen((5120, 2880), 2.0, 60.0), None);
        assert_eq!(
            sizes(&modes),
            vec![
                (4608, 2880),
                (5120, 2880),
                (3840, 2880),
                (2560, 1600),
                (1920, 1200),
                (1680, 1050),
                (1440, 900),
                (1280, 800),
                (3840, 2160),
                (2560, 1440),
                (1920, 1080),
                (1600, 900),
                (1280, 720),
                (2048, 1536),
                (1600, 1200),
                (1400, 1050),
                (1024, 768),
            ]
        );
    }

    /// A 1920×1080 1× screen: no standard size of its full 16:9 is listed twice
    #[test]
    fn offered_modes_for_a_1080p_screen() {
        let modes = offered_modes(&screen((1920, 1080), 1.0, 60.0), None);
        assert_eq!(
            sizes(&modes),
            vec![
                (1728, 1080),
                (1920, 1080),
                (1440, 1080),
                (1680, 1050),
                (1440, 900),
                (1280, 800),
                (1600, 900),
                (1280, 720),
                (1400, 1050),
                (1024, 768),
            ]
        );
    }

    /// Every mode is a multiple of 8 each way, below 65536, listed once, and there are at most
    /// 128
    #[test]
    fn offered_modes_are_whole_and_distinct() {
        for pixels in [
            (3456, 2234),
            (5120, 2880),
            (1920, 1080),
            (2560, 1664),
            (1366, 768),
            (800, 600),
            (100_000, 100_000),
            (7, 7),
            (0, 0),
        ] {
            let modes = offered_modes(&screen(pixels, 1.0, 60.0), None);
            assert!(modes.len() <= MAX_MODES);
            for (i, m) in modes.iter().enumerate() {
                assert!(m.width > 0 && m.height > 0, "{pixels:?}: {m:?}");
                let listed = modes[..i]
                    .iter()
                    .any(|n| (n.width, n.height) == (m.width, m.height));
                assert!(!listed, "{pixels:?}: {m:?} twice");
            }
            for m in modes.iter().take(3) {
                assert_eq!((m.width % 8, m.height % 8), (0, 0), "{pixels:?}: {m:?}");
                assert!(u32::from(m.width) <= pixels.0 && u32::from(m.height) <= pixels.1);
            }
        }
        assert!(offered_modes(&screen((0, 0), 1.0, 60.0), None).is_empty());
        let huge = offered_modes(&screen((100_000, 100_000), 1.0, 60.0), None);
        assert_eq!((huge[0].width, huge[0].height), (65528, 40952));
    }

    /// The refresh rate: the override, else the screen's, else 60 Hz
    #[test]
    fn the_refresh_rate_can_be_overridden() {
        let s = screen((3456, 2234), 2.0, 120.0);
        assert!(
            offered_modes(&s, Some(60.0))
                .iter()
                .all(|m| m.refresh_hz == 60.0)
        );
        assert!(
            offered_modes(&s, None)
                .iter()
                .all(|m| m.refresh_hz == 120.0)
        );
        let unknown = screen((1920, 1080), 1.0, 0.0);
        assert!(
            offered_modes(&unknown, None)
                .iter()
                .all(|m| m.refresh_hz == 60.0)
        );
        assert!(
            offered_modes(&unknown, Some(144.0))
                .iter()
                .all(|m| m.refresh_hz == 144.0)
        );
        assert!(
            offered_modes(&s, Some(f64::NAN))
                .iter()
                .all(|m| m.refresh_hz == 120.0)
        );
    }

    /// A Retina screen gives a HiDPI guest, a 1× screen a 1× one, sized at the screen's density
    #[test]
    fn the_density_is_the_screens() {
        let retina = density(&screen((3456, 2234), 2.0, 120.0), (3456, 2160));
        assert_eq!(retina.scale_factor, 2);
        // 220 ppi: 3456 × 25.46 / 220 and 2160 × 25.46 / 220.
        assert!((retina.size_in_millimeters.0 - 399.953_454_545_454_6).abs() < 1e-9);
        assert!((retina.size_in_millimeters.1 - 249.970_909_090_909_1).abs() < 1e-9);
        let one = density(&screen((1920, 1080), 1.0, 60.0), (1920, 1080));
        assert_eq!(one.scale_factor, 1);
        // 110 ppi: 1920 × 25.46 / 110 and 1080 × 25.46 / 110.
        assert!((one.size_in_millimeters.0 - 444.392_727_272_727_3).abs() < 1e-9);
        assert!((one.size_in_millimeters.1 - 249.970_909_090_909_1).abs() < 1e-9);
        // About the boundary: 141.9 ppi is 1×, 144.1 ppi 2×.
        assert_eq!(density(&screen((1, 1), 1.29, 60.0), (8, 8)).scale_factor, 1);
        assert_eq!(density(&screen((1, 1), 1.31, 60.0), (8, 8)).scale_factor, 2);
        // A scale that is not one is taken as 1×.
        assert_eq!(
            density(&screen((1, 1), f64::NAN, 60.0), (8, 8)).scale_factor,
            1
        );
    }

    /// A chosen 1920×1080 on a 120 Hz Retina screen: it first, then the standard sizes within
    /// it, at the screen's rate
    #[test]
    fn a_chosen_mode_comes_first_with_the_sizes_within_it() {
        let s = screen((3456, 2234), 2.0, 120.0);
        let modes = modes_up_to((1920, 1080), Some(&s), None);
        assert_eq!(
            sizes(&modes),
            vec![
                (1920, 1080),
                (1680, 1050),
                (1440, 900),
                (1280, 800),
                (1600, 900),
                (1280, 720),
                (1400, 1050),
                (1024, 768),
            ]
        );
        assert!(modes.iter().all(|m| m.refresh_hz == 120.0));
        assert!(
            modes_up_to((1920, 1080), Some(&s), Some(60.0))
                .iter()
                .all(|m| m.refresh_hz == 60.0)
        );
        assert_eq!(
            sizes(&modes_up_to((800, 600), Some(&s), None)),
            vec![(800, 600)]
        );
        assert!(modes_up_to((0, 1080), Some(&s), None).is_empty());
    }

    /// With no screen to match, a chosen mode is offered as on a screen, at 60 Hz unless the
    /// rate is overridden
    #[test]
    fn a_chosen_mode_without_a_screen_runs_at_60_hz() {
        let s = screen((3456, 2234), 2.0, 120.0);
        let on_screen = modes_up_to((3840, 2400), Some(&s), None);
        let modes = modes_up_to((3840, 2400), None, None);
        assert_eq!(sizes(&modes), sizes(&on_screen));
        assert!(modes.iter().all(|m| m.refresh_hz == DEFAULT_REFRESH_HZ));
        assert!(
            modes_up_to((3840, 2400), None, Some(144.0))
                .iter()
                .all(|m| m.refresh_hz == 144.0)
        );
        assert!(
            modes_up_to((3840, 2400), None, Some(f64::NAN))
                .iter()
                .all(|m| m.refresh_hz == DEFAULT_REFRESH_HZ)
        );
    }

    /// A chosen mode is told 110 ppi on any screen: a 1× guest desktop
    #[test]
    fn a_chosen_mode_is_1x() {
        let d = density_1x((1920, 1080));
        assert_eq!(d.scale_factor, 1);
        assert!((d.size_in_millimeters.0 - 444.392_727_272_727_3).abs() < 1e-9);
        assert!((d.size_in_millimeters.1 - 249.970_909_090_909_1).abs() < 1e-9);
        assert_eq!(density_1x((3456, 2160)).scale_factor, 1);
    }

    /// A chosen mode told a chosen density: 2560×1600 at 200 ppi is sized at 200 ppi with a
    /// scale factor of 2, and the guest runs it at 2×; no density, or one that is not a positive
    /// finite number, is 110 ppi
    #[test]
    fn a_chosen_mode_at_a_chosen_ppi() {
        let d = density_chosen((2560, 1600), Some(200.0));
        assert_eq!(d.scale_factor, 2);
        // 2560 × 25.46 / 200 and 1600 × 25.46 / 200.
        assert!((d.size_in_millimeters.0 - 325.888).abs() < 1e-9);
        assert!((d.size_in_millimeters.1 - 203.68).abs() < 1e-9);
        assert_eq!(guest_scale((2560, 1600), d.size_in_millimeters), 2);
        // 130 ppi is below the HiDPI boundary.
        let low = density_chosen((1920, 1080), Some(130.0));
        assert_eq!(low.scale_factor, 1);
        assert_eq!(guest_scale((1920, 1080), low.size_in_millimeters), 1);
        for ppi in [
            None,
            Some(0.0),
            Some(-200.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert_eq!(
                density_chosen((1920, 1080), ppi),
                density_1x((1920, 1080)),
                "{ppi:?}"
            );
        }
    }

    /// A chosen density tells the scale the guest runs the mode at, not 2 above 143 ppi:
    /// 2560×1600 told 146 ppi reads as 45 cm, 144.5 ppi, and is 1×; told 148, 44 cm and 2×
    #[test]
    fn a_chosen_density_tells_the_scale_the_guest_runs() {
        let at_146 = density_chosen((2560, 1600), Some(146.0));
        assert_eq!(at_146.scale_factor, 1);
        assert_eq!(
            at_146.size_in_millimeters,
            density_at(146.0, (2560, 1600)).size_in_millimeters
        );
        assert_eq!(density_chosen((2560, 1600), Some(148.0)).scale_factor, 2);
        for mode in [
            (2560, 1600),
            (1920, 1080),
            (3456, 2160),
            (2021, 1263),
            (800, 600),
        ] {
            for ppi in 1..=400 {
                let d = density_chosen(mode, Some(f64::from(ppi)));
                assert_eq!(
                    d.scale_factor,
                    guest_scale(mode, d.size_in_millimeters),
                    "{mode:?} at {ppi}"
                );
            }
        }
    }

    /// The densities a macOS 13.6 guest was booted at, and the scale it ran: it reads the width
    /// in whole centimetres, rounded to whole millimetres and then up, and runs HiDPI from 2021
    /// pixels across 35 cm on, but not 1280 pixels wide
    #[test]
    fn the_guest_scale_follows_its_reading_in_whole_centimetres() {
        // (mode, told ppi, whole centimetres read, scale run)
        let measured = [
            ((3456, 2160), 144.0, 62.0, 1),
            ((2560, 1600), 146.0, 45.0, 1),
            ((3456, 2160), 147.0, 60.0, 1),
            // 590.54 mm: 591, then 60 cm (146.30 ppi), where 590 would be 59 (148.8). Its
            // 2160 pixels across 37 cm come to 148.3 ppi: the height does not count.
            ((3456, 2160), 149.0, 60.0, 1),
            ((2944, 1840), 148.0, 51.0, 1),
            ((2656, 1660), 149.0, 46.0, 1),
            ((1790, 1119), 148.0, 31.0, 1),
            ((3811, 2382), 148.0, 66.0, 1),
            ((2021, 1263), 149.0, 35.0, 2),
            ((3176, 1984), 148.0, 55.0, 2),
            ((2080, 1300), 149.0, 36.0, 2),
            ((3008, 1880), 148.0, 52.0, 2),
            // 440.39 mm: 440, then 44 cm (147.8 ppi), where 45 would be 144.5.
            ((2560, 1600), 148.0, 44.0, 2),
            ((2560, 1600), 149.0, 44.0, 2),
            ((1920, 1200), 149.0, 33.0, 2),
            ((2624, 1640), 149.0, 45.0, 2),
            // 1536 pixels across 27 cm come to 144.5 ppi.
            ((2048, 1536), 149.0, 35.0, 2),
            ((3456, 2160), 220.0, 40.0, 2),
            ((3456, 2160), 254.0, 35.0, 2),
            ((5120, 2880), 218.0, 60.0, 2),
            ((2560, 1600), 227.0, 29.0, 2),
            // 216.7 ppi, but 1280 pixels wide: HiDPI would look like 640×400.
            ((1280, 800), 220.0, 15.0, 1),
        ];
        for (mode, ppi, centimeters, scale) in measured {
            let size = density_at(ppi, mode).size_in_millimeters;
            assert_eq!(
                guest_centimeters(size.0),
                Some(centimeters),
                "{mode:?} at {ppi}"
            );
            assert_eq!(guest_scale(mode, size), scale, "{mode:?} at {ppi}");
        }
        // Whole millimetres to nearest, then whole centimetres up.
        assert_eq!(guest_centimeters(440.49), Some(44.0));
        assert_eq!(guest_centimeters(440.5), Some(45.0));
        assert_eq!(guest_centimeters(430.5), Some(44.0));
        assert_eq!(guest_centimeters(0.6), Some(1.0));
        for none in [0.4, 0.0, -1.0, f64::NAN, f64::INFINITY, 1e300] {
            assert_eq!(guest_centimeters(none), None, "{none}");
        }
    }

    /// The guest's scale follows the density its mode has on the size the display was told
    #[test]
    fn the_guest_scale_follows_the_told_density() {
        let retina = density(&screen((3456, 2234), 2.0, 120.0), (3456, 2160));
        assert_eq!(guest_scale((3456, 2160), retina.size_in_millimeters), 2);
        // A smaller mode on the same size is below 143 ppi.
        assert_eq!(guest_scale((1920, 1200), retina.size_in_millimeters), 1);
        let chosen = density_1x((1920, 1080));
        assert_eq!(guest_scale((1920, 1080), chosen.size_in_millimeters), 1);
        // QEMU's 400 × 300 mm without a window's policy.
        assert_eq!(guest_scale((1920, 1080), (400.0, 300.0)), 1);
        assert_eq!(guest_scale((3456, 2160), (400.0, 300.0)), 2);
        assert_eq!(guest_scale((1920, 1080), (0.0, 0.0)), 1);
        assert_eq!(guest_scale((1920, 1080), (f64::NAN, 1.0)), 1);
    }
}
