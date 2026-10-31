//! The window's control protocol: what a client of the control socket (Linux's
//! `REIMS_VGPU_WINDOW_CONTROL`, `native/linux/control.rs`) asks of the windows, and what they
//! tell it. Lines of UTF-8 text, one message each, ended by a newline; words are separated by
//! spaces.
//!
//! The client asks ([`Request`]):
//! - `toolbar <atlas> <button-px> <id>,<id>,...`: show the button bar ([`crate::decor`]), its
//!   buttons named by the ids in order, their images in the raw RGBA8 file at path `<atlas>`
//!   (`2 * button-px` by `n * button-px` pixels, [`crate::decor::Atlas`]); `toolbar off` hides
//!   it. The path is the rest of the line but its last two words, so it may hold spaces.
//! - `rotate <degrees>`: turn the guest's image 0, 90, 180 or 270 degrees clockwise.
//!
//! The windows tell it ([`button_line`]): `button <id>`, when a button of the bar is clicked (the
//! pointer's button released over the button it was pressed on).
//!
//! A line that is none of these is logged and ignored.

use std::path::PathBuf;

use crate::decor::Rotation;

/// What a client asks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    /// Show the bar: its atlas's path, a button's side in pixels, and the buttons' ids.
    Toolbar {
        /// The atlas's file.
        atlas: PathBuf,
        /// A button's side, in the atlas's pixels and the window's points.
        cell: u32,
        /// The buttons' ids, in the atlas's order.
        ids: Vec<String>,
    },
    /// Hide the bar.
    ToolbarOff,
    /// Turn the image.
    Rotate(Rotation),
}

/// The request `line` makes, a newline or carriage return at its end ignored, or why it is
/// none.
pub fn parse(line: &str) -> Result<Request, String> {
    let line = line.trim_end_matches(['\n', '\r']).trim();
    let (verb, rest) = line.split_once(' ').unwrap_or((line, ""));
    let rest = rest.trim();
    match verb {
        "toolbar" if rest == "off" => Ok(Request::ToolbarOff),
        "toolbar" => {
            let (rest, ids) = rest
                .rsplit_once(' ')
                .ok_or("toolbar: <atlas> <button-px> <id>,<id>,...")?;
            let (atlas, cell) = rest
                .trim_end()
                .rsplit_once(' ')
                .ok_or("toolbar: <atlas> <button-px> <id>,<id>,...")?;
            let atlas = atlas.trim();
            if atlas.is_empty() {
                return Err("toolbar: no atlas".into());
            }
            let cell = cell
                .parse::<u32>()
                .map_err(|_| format!("toolbar: {cell:?} is not a button size in pixels"))?;
            let ids: Vec<String> = ids.split(',').map(str::to_owned).collect();
            if ids.iter().any(String::is_empty) {
                return Err(format!("toolbar: {:?} names an empty id", ids.join(",")));
            }
            Ok(Request::Toolbar {
                atlas: PathBuf::from(atlas),
                cell,
                ids,
            })
        }
        "rotate" => rest
            .parse::<u32>()
            .ok()
            .and_then(Rotation::from_degrees)
            .map(Request::Rotate)
            .ok_or_else(|| format!("rotate: {rest:?} is not 0, 90, 180 or 270")),
        _ => Err(format!("{verb:?} is no request")),
    }
}

/// The line telling the client button `id` was clicked, its newline included.
pub fn button_line(id: &str) -> String {
    format!("button {id}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toolbar request names the atlas, the button size and the ids; its path may hold
    /// spaces, and `toolbar off` hides the bar
    #[test]
    fn a_toolbar_request_names_the_atlas_size_and_ids() {
        assert_eq!(
            parse("toolbar /tmp/atlas.rgba 48 home,lock,rotate\n"),
            Ok(Request::Toolbar {
                atlas: PathBuf::from("/tmp/atlas.rgba"),
                cell: 48,
                ids: vec!["home".into(), "lock".into(), "rotate".into()],
            })
        );
        assert_eq!(
            parse("toolbar /tmp/my atlas.rgba 32 home\r\n"),
            Ok(Request::Toolbar {
                atlas: PathBuf::from("/tmp/my atlas.rgba"),
                cell: 32,
                ids: vec!["home".into()],
            })
        );
        assert_eq!(parse("toolbar off"), Ok(Request::ToolbarOff));
        assert_eq!(parse("  toolbar off \n"), Ok(Request::ToolbarOff));
    }

    /// A toolbar request with a missing part, a size that is not a number or an empty id is
    /// refused
    #[test]
    fn a_malformed_toolbar_request_is_refused() {
        for bad in [
            "toolbar",
            "toolbar /tmp/a.rgba",
            "toolbar /tmp/a.rgba 48",
            "toolbar 48 home",
            "toolbar /tmp/a.rgba big home",
            "toolbar /tmp/a.rgba -4 home",
            "toolbar /tmp/a.rgba 48 home,,lock",
            "toolbar /tmp/a.rgba 48 home,",
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }

    /// A rotate request takes the four quarter turns, and nothing else
    #[test]
    fn a_rotate_request_takes_the_quarter_turns() {
        assert_eq!(parse("rotate 0"), Ok(Request::Rotate(Rotation::None)));
        assert_eq!(parse("rotate 90\n"), Ok(Request::Rotate(Rotation::Quarter)));
        assert_eq!(parse("rotate 180"), Ok(Request::Rotate(Rotation::Half)));
        assert_eq!(
            parse("rotate 270"),
            Ok(Request::Rotate(Rotation::ThreeQuarters))
        );
        for bad in [
            "rotate",
            "rotate 45",
            "rotate -90",
            "rotate 360",
            "rotate ninety",
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }

    /// Any other line is no request; a click is told as `button <id>` and a newline
    #[test]
    fn other_lines_are_refused_and_a_click_is_one_line() {
        for bad in ["", "\n", "hello", "Rotate 90", "toolbaroff"] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(button_line("home"), "button home\n");
    }
}
