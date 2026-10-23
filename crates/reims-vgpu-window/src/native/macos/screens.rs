//! The host's screens, from CoreGraphics, which answers on any thread: "screen i" is the i-th
//! display `CGGetActiveDisplayList` lists, the main display first. The main thread finds a
//! screen's `NSScreen` by its display ID (`deviceDescription["NSScreenNumber"]`).
//!
//! A screen's size is its current mode's pixels (`CGDisplayModeGetPixelWidth`), and its scale
//! those pixels over the mode's points (`CGDisplayModeGetWidth`). Its refresh rate is its current
//! mode's `CGDisplayModeGetRefreshRate`; where that is 0, unknown, the highest among the modes of
//! the same pixel size stands in, and a screen whose rates are all unknown runs at 60 Hz
//! (`screens::screen_refresh`).
//!
//! Measured on the M3 Max's built-in ProMotion screen (3456×2234 pixels, 1728×1117 points,
//! refresh set to ProMotion), macOS 26.7, and again on macOS 27.0.1 (26A434) with the displays
//! asleep and the session locked:
//! - the current mode reports 120 Hz, not 0, and the modes of its pixel size are listed at
//!   120, 60, 59.94, 50, 48 and 47.95 Hz, each a mode of its own (I/O Kit mode IDs 54 to 59 at
//!   1728×1117 points), so a rate the person sets below ProMotion's is another current mode, at
//!   that rate (read from the list; the setting was not changed to see it). `NSScreen`'s
//!   `maximumFramesPerSecond` is 120 (`minimumRefreshInterval` 1/120 s, `maximumRefreshInterval`
//!   1/24 s), and CoreVideo's display link's nominal rate 120 Hz;
//! - `CGDisplayCopyAllDisplayModes` lists the current, scaled mode (1728×1117 points) only when
//!   asked for `kCGDisplayShowDuplicateLowResolutionModes`; without it only the 1:1 modes of
//!   that pixel size appear, at the same rates. The modes are asked for with it;
//! - while the displays sleep, `CGGetActiveDisplayList` lists none, and
//!   `CGGetOnlineDisplayList` still lists the screen with its current mode. So when no display
//!   is active, the online displays that mirror none stand in, and a host started with its
//!   screens asleep is still offered modes matching them.

use objc2_core_foundation::{CFArray, CFBoolean, CFDictionary, CFRetained, CFString};
use objc2_core_graphics::{
    CGDirectDisplayID, CGDisplayCopyAllDisplayModes, CGDisplayCopyDisplayMode,
    CGDisplayMirrorsDisplay, CGDisplayMode, CGError, CGGetActiveDisplayList,
    CGGetOnlineDisplayList, CGMainDisplayID, kCGDisplayShowDuplicateLowResolutionModes,
    kCGNullDirectDisplay,
};

use crate::screens::{HostScreen, screen_refresh};

/// The most displays asked for.
const MAX_DISPLAYS: usize = 32;

/// A CoreGraphics display list: `list(max, displays, count)`.
type DisplayList = unsafe extern "C-unwind" fn(u32, *mut CGDirectDisplayID, *mut u32) -> CGError;

/// The displays `list` gives, in its order. Empty on an error.
fn display_list(list: DisplayList) -> Vec<CGDirectDisplayID> {
    let mut ids = [kCGNullDirectDisplay; MAX_DISPLAYS];
    let mut count = 0u32;
    // SAFETY: `ids` holds `MAX_DISPLAYS` IDs, the most the call is told to write, and `count`
    // is a u32 it writes.
    let error = unsafe { list(MAX_DISPLAYS as u32, ids.as_mut_ptr(), &mut count) };
    if error != CGError::Success {
        return Vec::new();
    }
    ids.into_iter().take(count as usize).collect()
}

/// The displays screens are taken from: the active ones, the main display first; where none is
/// active (every display asleep), the online ones that mirror no other.
fn displays() -> Vec<CGDirectDisplayID> {
    let mut ids = display_list(CGGetActiveDisplayList);
    if ids.is_empty() {
        ids = display_list(CGGetOnlineDisplayList);
        ids.retain(|&id| CGDisplayMirrorsDisplay(id) == kCGNullDirectDisplay);
    }
    let main = CGMainDisplayID();
    if let Some(at) = ids.iter().position(|&id| id == main) {
        let id = ids.remove(at);
        ids.insert(0, id);
    }
    ids
}

/// Every mode of `display`, the scaled ones included.
fn all_modes(display: CGDirectDisplayID) -> Vec<CFRetained<CGDisplayMode>> {
    // SAFETY: `kCGDisplayShowDuplicateLowResolutionModes` is a constant CoreGraphics defines.
    let key: &CFString = unsafe { kCGDisplayShowDuplicateLowResolutionModes };
    let options = CFDictionary::<CFString, CFBoolean>::from_slices(&[key], &[CFBoolean::new(true)]);
    // SAFETY: the options dictionary maps CFStrings to CFBooleans, as the call reads it.
    let Some(modes) = (unsafe { CGDisplayCopyAllDisplayModes(display, Some(options.as_opaque())) })
    else {
        return Vec::new();
    };
    // SAFETY: the array CoreGraphics returns holds CGDisplayModes.
    let modes: &CFArray<CGDisplayMode> = unsafe { modes.cast_unchecked() };
    (0..modes.len()).filter_map(|i| modes.get(i)).collect()
}

/// The screen `display` is, or `None` where it has no current mode.
fn screen(display: CGDirectDisplayID) -> Option<HostScreen> {
    let mode = CGDisplayCopyDisplayMode(display)?;
    let mode = Some(&*mode);
    let pixels = mode_pixels(mode);
    let points = CGDisplayMode::width(mode);
    let scale = if points > 0 && pixels.0 > 0 {
        f64::from(pixels.0) / points as f64
    } else {
        1.0
    };
    let modes = all_modes(display).into_iter().map(|other| {
        let other = Some(&*other);
        (mode_pixels(other), CGDisplayMode::refresh_rate(other))
    });
    Some(HostScreen {
        id: display,
        pixels,
        scale,
        refresh_hz: screen_refresh(pixels, CGDisplayMode::refresh_rate(mode), modes),
    })
}

/// `mode`'s size in pixels, width then height.
fn mode_pixels(mode: Option<&CGDisplayMode>) -> (u32, u32) {
    let side = |pixels: usize| u32::try_from(pixels).unwrap_or(u32::MAX);
    (
        side(CGDisplayMode::pixel_width(mode)),
        side(CGDisplayMode::pixel_height(mode)),
    )
}

/// The host's screens, the main one first. Empty where CoreGraphics lists none.
pub(crate) fn host_screens() -> Vec<HostScreen> {
    displays().into_iter().filter_map(screen).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screens::screen_for_port;

    /// CoreGraphics' screens come main first, each with pixels, a scale and a rate
    #[test]
    fn host_screens_come_main_first() {
        let screens = host_screens();
        if let Some(first) = screens.first() {
            assert_eq!(first.id, CGMainDisplayID());
        }
        for s in &screens {
            assert!(s.pixels.0 > 0 && s.pixels.1 > 0, "{s:?}");
            assert!(s.scale.is_finite() && s.scale > 0.0, "{s:?}");
            assert!(s.refresh_hz.is_finite() && s.refresh_hz > 0.0, "{s:?}");
        }
        assert_eq!(screen_for_port(0).is_some(), !screens.is_empty());
    }
}
