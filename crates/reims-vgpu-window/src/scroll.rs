//! Wheel and trackpad deltas to the guest's scrolling: wheel notches, or smooth scrolling where
//! the host takes it (a [`ScrollFn`](crate::config::ScrollFn)).
//!
//! As notches ([`Scroll`]): a notched mouse wheel gives the guest one notch per notch, and a
//! precise device (a trackpad, a Magic Mouse) one notch per [`POINTS_PER_NOTCH`] points of
//! movement, momentum included, counted afresh at each new gesture and at each change of
//! direction.
//!
//! Smoothly ([`smooth`], [`Gesture`]): the guest gets each event's distance in points, with the
//! phases of its gesture and of the momentum scrolling after it, and scrolls as the host does. A
//! gesture that begins over the guest's image goes to the guest to its end, momentum included,
//! wherever the pointer goes meanwhile; one that begins elsewhere does not. A gesture begins where
//! the pointer is at its first scroll (`Began`), or, with the pointer still where it was, where
//! the fingers that began it rested on the device first (`MayBegin`). An end the window never
//! sees (the fingers' `Cancelled`, the gesture's `Ended`) leaves nothing to the next gesture: a
//! pointer that has moved since the fingers rested shows they lifted, and a second `Began` is a
//! gesture of its own.
//!
//! The guest gets the device's own direction. With the host's natural scrolling on, macOS
//! reverses the deltas it reports and says so (`isDirectionInvertedFromDevice`); they are
//! reversed back, so that the guest's own natural-scrolling setting is the one that applies.

use crate::config::{MomentumPhase, ScrollPhase};

/// Points of precise scrolling per guest notch.
pub const POINTS_PER_NOTCH: f64 = 40.0;

/// Points a line of a notched wheel's event stands for: macOS gives a wheel event's point delta
/// (`kCGScrollWheelEventPointDeltaAxis1`) as its line delta (`scrollingDeltaY`, accelerated)
/// times 10, rounded: 2.307 lines are 24 points.
pub const POINTS_PER_LINE: f64 = 10.0;

/// One scroll event, as the host reported it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScrollInput {
    /// Whether the device is precise (`NSEvent.hasPreciseScrollingDeltas`).
    pub precise: bool,
    /// `(x, y)`: `NSEvent.scrollingDeltaX` and `scrollingDeltaY`, points for a precise device.
    pub delta: (f64, f64),
    /// `(x, y)`: the event's whole lines, its `CGEvent`'s `kCGScrollWheelEventDeltaAxis2` and
    /// `kCGScrollWheelEventDeltaAxis1`, for a notched wheel.
    pub lines: (i64, i64),
    /// `NSEvent.isDirectionInvertedFromDevice`: the deltas are reversed from the device's.
    pub inverted: bool,
    /// `NSEvent.phase`.
    pub phase: ScrollPhase,
    /// `NSEvent.momentumPhase`.
    pub momentum: MomentumPhase,
}

impl ScrollInput {
    /// Whether a gesture, or its momentum, begins: `phase` or `momentum` is Began.
    pub fn began(&self) -> bool {
        self.phase == ScrollPhase::Began || self.momentum == MomentumPhase::Began
    }
}

/// Precise scrolling's points not yet a whole notch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scroll {
    /// `(x, y)` points, in the device's direction.
    remainder: (f64, f64),
}

/// The notches in `points`, and the points left over.
fn notches(points: f64) -> (i32, f64) {
    let whole = (points / POINTS_PER_NOTCH).trunc();
    let left = points - whole * POINTS_PER_NOTCH;
    (
        whole.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32,
        left,
    )
}

/// `value` where it is finite, else 0.
fn finite(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

/// `remainder` and `points` added, counting afresh when they point opposite ways.
fn accumulate(remainder: f64, points: f64) -> f64 {
    if remainder * points < 0.0 {
        points
    } else {
        remainder + points
    }
}

impl Scroll {
    /// No points counted.
    pub fn new() -> Scroll {
        Scroll::default()
    }

    /// The guest's notches for one event: `(dy, dx)`, `dy` above 0 up (away from the person),
    /// `dx` above 0 left, in the device's own direction.
    pub fn feed(&mut self, input: ScrollInput) -> (i32, i32) {
        let sign = if input.inverted { -1.0 } else { 1.0 };
        if !input.precise {
            self.remainder = (0.0, 0.0);
            let lines = |n: i64| -> i32 {
                let n = if input.inverted {
                    n.saturating_neg()
                } else {
                    n
                };
                n.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
            };
            return (lines(input.lines.1), lines(input.lines.0));
        }
        if input.began() {
            self.remainder = (0.0, 0.0);
        }
        let x = accumulate(self.remainder.0, sign * finite(input.delta.0));
        let y = accumulate(self.remainder.1, sign * finite(input.delta.1));
        let (dx, x) = notches(x);
        let (dy, y) = notches(y);
        self.remainder = (x, y);
        (dy, dx)
    }
}

/// `points` in 1/256 point, rounded; 0 for what is not a number.
fn fixed(points: f64) -> i32 {
    (finite(points) * 256.0)
        .round()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

/// One event as smooth scrolling: `(dy, dx)` in 1/256 point, `dy` above 0 up (away from the
/// person), `dx` above 0 left, in the device's own direction. A precise device's deltas are
/// points; a notched wheel's are lines, [`POINTS_PER_LINE`] points each.
pub fn smooth(input: ScrollInput) -> (i32, i32) {
    let sign = if input.inverted { -1.0 } else { 1.0 };
    let scale = if input.precise { 1.0 } else { POINTS_PER_LINE };
    (
        fixed(sign * scale * finite(input.delta.1)),
        fixed(sign * scale * finite(input.delta.0)),
    )
}

/// Where smooth scrolling is in a gesture: which of its events go to the guest.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Gesture {
    /// In no gesture.
    #[default]
    Idle,
    /// Fingers rest on the device and may begin a gesture, the pointer at `at` (in the view's
    /// points, where known), over the image (`over`) or not: their `MayBegin` went to the guest
    /// where it was over the image, and so does their `Cancelled`.
    MayBegin {
        /// Whether the pointer was over the image.
        over: bool,
        /// Where the pointer was.
        at: Option<(f64, f64)>,
    },
    /// A gesture that began over the image: its events go to the guest.
    Open,
    /// That gesture ended: the momentum that may follow goes to the guest.
    Ended,
    /// That gesture's momentum, which goes to the guest to its end.
    Momentum,
    /// A gesture that began off the image, whose events do not go to the guest.
    Elsewhere,
}

impl Gesture {
    /// Whether an event with `phase` and `momentum` goes to the guest, the pointer being at `at`
    /// in the view's points (where known) and over the image (`over`) or not. An event of no
    /// phase goes where the pointer is over the image; an event of a gesture, where the gesture
    /// began over it (the module doc).
    pub fn route(
        &mut self,
        phase: ScrollPhase,
        momentum: MomentumPhase,
        at: Option<(f64, f64)>,
        over: bool,
    ) -> bool {
        use Gesture::{Elsewhere, Ended, Idle, MayBegin, Momentum, Open};
        // Fingers that rested with the pointer where it still is begin the gesture where they
        // rested. A pointer that moved since shows they lifted, their Cancelled unseen: the
        // gesture begins where the pointer is now, as after any other state, a gesture whose
        // Ended went unseen included.
        let begins_over = match *self {
            MayBegin {
                over: rested,
                at: was,
            } if at.is_some() && at == was => rested,
            _ => over,
        };
        let (next, send) = match (phase, momentum, *self) {
            (ScrollPhase::MayBegin, _, _) => (MayBegin { over, at }, over),
            (ScrollPhase::Began, _, _) if begins_over => (Open, true),
            (ScrollPhase::Began, _, _) => (Elsewhere, false),
            // Resting fingers held still (Stationary) go where they would begin.
            (ScrollPhase::Changed, _, MayBegin { .. }) => (*self, begins_over),
            (ScrollPhase::Changed, _, _) => (*self, *self == Open),
            (ScrollPhase::Ended, _, Open) => (Ended, true),
            (ScrollPhase::Cancelled, _, Open | MayBegin { over: true, .. }) => (Idle, true),
            (ScrollPhase::Ended | ScrollPhase::Cancelled, _, _) => (Idle, false),
            (ScrollPhase::None, MomentumPhase::Began, Ended) => (Momentum, true),
            (ScrollPhase::None, MomentumPhase::Began, _) => (Idle, false),
            (ScrollPhase::None, MomentumPhase::Changed, _) => (*self, *self == Momentum),
            (ScrollPhase::None, MomentumPhase::Ended, _) => (Idle, *self == Momentum),
            (ScrollPhase::None, MomentumPhase::None, _) => (*self, over),
        };
        *self = next;
        send
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wheel(x: i64, y: i64, inverted: bool) -> ScrollInput {
        ScrollInput {
            precise: false,
            delta: (x as f64 * 0.1, y as f64 * 0.1),
            lines: (x, y),
            inverted,
            ..ScrollInput::default()
        }
    }

    fn precise(x: f64, y: f64, began: bool) -> ScrollInput {
        ScrollInput {
            precise: true,
            delta: (x, y),
            lines: (0, 0),
            inverted: false,
            phase: if began {
                ScrollPhase::Began
            } else {
                ScrollPhase::Changed
            },
            momentum: MomentumPhase::None,
        }
    }

    /// A notched wheel gives one notch per line, both ways and both axes
    #[test]
    fn a_notched_wheel_gives_a_notch_per_line() {
        let mut scroll = Scroll::new();
        assert_eq!(scroll.feed(wheel(0, 1, false)), (1, 0));
        assert_eq!(scroll.feed(wheel(0, -1, false)), (-1, 0));
        assert_eq!(scroll.feed(wheel(0, 3, false)), (3, 0));
        assert_eq!(scroll.feed(wheel(2, 0, false)), (0, 2));
        assert_eq!(scroll.feed(wheel(-1, 0, false)), (0, -1));
        assert_eq!(scroll.feed(wheel(0, i64::MAX, false)), (i32::MAX, 0));
    }

    /// A precise device gives a notch per 40 points and keeps the rest
    #[test]
    fn precise_scrolling_gives_a_notch_per_40_points() {
        let mut scroll = Scroll::new();
        assert_eq!(scroll.feed(precise(0.0, 25.0, true)), (0, 0));
        assert_eq!(scroll.feed(precise(0.0, 25.0, false)), (1, 0));
        assert_eq!(scroll.feed(precise(0.0, 30.0, false)), (1, 0));
        assert_eq!(scroll.feed(precise(0.0, 90.0, false)), (2, 0));
        // 25 + 25 + 30 + 90 = 170 = 4 notches and 10 points: 30 more make a fifth.
        assert_eq!(scroll.feed(precise(0.0, 29.0, false)), (0, 0));
        assert_eq!(scroll.feed(precise(0.0, 1.0, false)), (1, 0));
        // Momentum events are precise events like any other.
        assert_eq!(scroll.feed(precise(0.0, 40.0, false)), (1, 0));
        // The x axis counts on its own.
        assert_eq!(scroll.feed(precise(-50.0, 0.0, false)), (0, -1));
        assert_eq!(scroll.feed(precise(-30.0, 0.0, false)), (0, -1));
        // What is not a number counts nothing.
        assert_eq!(scroll.feed(precise(f64::NAN, f64::INFINITY, false)), (0, 0));
    }

    /// A new gesture (or its momentum) counts afresh
    #[test]
    fn a_new_gesture_restarts_the_count() {
        let mut scroll = Scroll::new();
        assert_eq!(scroll.feed(precise(0.0, 39.0, true)), (0, 0));
        assert_eq!(scroll.feed(precise(0.0, 39.0, true)), (0, 0));
        assert_eq!(scroll.feed(precise(0.0, 1.0, false)), (1, 0));
        // A notched event drops what a precise one left.
        assert_eq!(scroll.feed(precise(0.0, 39.0, false)), (0, 0));
        assert_eq!(scroll.feed(wheel(0, 0, false)), (0, 0));
        assert_eq!(scroll.feed(precise(0.0, 1.0, false)), (0, 0));
    }

    /// Turning back counts afresh from the turn
    #[test]
    fn a_direction_change_restarts_the_count() {
        let mut scroll = Scroll::new();
        assert_eq!(scroll.feed(precise(0.0, 39.0, true)), (0, 0));
        assert_eq!(scroll.feed(precise(0.0, -39.0, false)), (0, 0));
        assert_eq!(scroll.feed(precise(0.0, -1.0, false)), (-1, 0));
        assert_eq!(scroll.feed(precise(30.0, 0.0, false)), (0, 0));
        assert_eq!(scroll.feed(precise(-30.0, 0.0, false)), (0, 0));
        assert_eq!(scroll.feed(precise(-10.0, 0.0, false)), (0, -1));
    }

    /// Reversed deltas (the host's natural scrolling) are reversed back
    #[test]
    fn the_guest_gets_the_physical_direction() {
        let mut scroll = Scroll::new();
        // The wheel turned up, natural scrolling on: reported down, sent up.
        assert_eq!(scroll.feed(wheel(0, -1, true)), (1, 0));
        assert_eq!(scroll.feed(wheel(1, 0, true)), (0, -1));
        assert_eq!(scroll.feed(wheel(0, i64::MIN, true)), (i32::MAX, 0));
        // Fingers moved up on a trackpad, natural scrolling on.
        let mut input = precise(0.0, -80.0, true);
        input.inverted = true;
        assert_eq!(scroll.feed(input), (2, 0));
        let mut input = precise(-45.0, 0.0, true);
        input.inverted = true;
        assert_eq!(scroll.feed(input), (0, 1));
    }

    /// A precise device's points, and a notched wheel's lines at 10 points each, go in 1/256
    /// point, in the device's direction
    #[test]
    fn smooth_scrolling_is_in_256ths_of_a_point() {
        assert_eq!(smooth(precise(0.0, 6.0, false)), (1536, 0));
        assert_eq!(smooth(precise(-2.5, 0.25, false)), (64, -640));
        // Natural scrolling on: fingers up are reported down, and sent up.
        let mut input = precise(3.0, -6.0, false);
        input.inverted = true;
        assert_eq!(smooth(input), (1536, -768));
        // A wheel's accelerated lines, 10 points each.
        let input = ScrollInput {
            delta: (0.0, 2.307),
            lines: (0, 2),
            ..ScrollInput::default()
        };
        assert_eq!(smooth(input), (5906, 0));
        assert_eq!(smooth(wheel(0, -1, true)), (256, 0));
        // What is not a number is 0, and what does not fit is held at the end.
        assert_eq!(smooth(precise(f64::NAN, f64::INFINITY, false)), (0, 0));
        assert_eq!(smooth(precise(-1e12, 1e12, false)), (i32::MAX, i32::MIN));
    }

    /// A gesture that begins over the image goes to the guest to its end, momentum included,
    /// wherever the pointer goes
    #[test]
    fn a_gesture_begun_over_the_image_is_sent_to_its_end() {
        use MomentumPhase as M;
        use ScrollPhase as P;
        let (here, there) = (Some((600.0, 300.0)), Some((20.0, 300.0)));
        let mut g = Gesture::default();
        // Fingers rest over the image, and scroll with the pointer still there, the image gone
        // from under it.
        assert!(g.route(P::MayBegin, M::None, here, true));
        assert!(g.route(P::Began, M::None, here, false));
        assert!(g.route(P::Changed, M::None, there, false));
        assert!(g.route(P::Ended, M::None, there, false));
        assert!(g.route(P::None, M::Began, there, false));
        assert!(g.route(P::None, M::Changed, there, false));
        assert!(g.route(P::None, M::Ended, there, false));
        assert_eq!(g, Gesture::Idle);
        // Without MayBegin, and without momentum.
        assert!(g.route(P::Began, M::None, here, true));
        assert!(g.route(P::Ended, M::None, there, false));
        // Fingers that may have begun, and lift.
        assert!(g.route(P::MayBegin, M::None, here, true));
        assert!(g.route(P::Cancelled, M::None, there, false));
        assert_eq!(g, Gesture::Idle);
    }

    /// A gesture that begins off the image is not sent, nor is its momentum, even over the image
    #[test]
    fn a_gesture_begun_off_the_image_is_not_sent() {
        use MomentumPhase as M;
        use ScrollPhase as P;
        let (here, there) = (Some((20.0, 300.0)), Some((600.0, 300.0)));
        let mut g = Gesture::default();
        assert!(!g.route(P::MayBegin, M::None, here, false));
        assert!(!g.route(P::Began, M::None, here, true));
        assert!(!g.route(P::Changed, M::None, there, true));
        assert!(!g.route(P::Ended, M::None, there, true));
        assert!(!g.route(P::None, M::Began, there, true));
        assert!(!g.route(P::None, M::Changed, there, true));
        assert!(!g.route(P::None, M::Ended, there, true));
        // Momentum after a gesture that ended long ago is not the guest's either.
        assert!(g.route(P::Began, M::None, there, true));
        assert!(g.route(P::Ended, M::None, there, true));
        assert!(!g.route(P::Began, M::None, here, false));
        assert!(!g.route(P::None, M::Began, there, true));
        assert_eq!(g, Gesture::Idle);
    }

    /// An end the window never saw leaves nothing to the next gesture: fingers that rested over
    /// the image, or off it, and lifted unseen (the pointer has moved since) do not decide where
    /// the next gesture goes, nor does a gesture whose Ended went unseen
    #[test]
    fn an_end_unseen_leaves_nothing_to_the_next_gesture() {
        use MomentumPhase as M;
        use ScrollPhase as P;
        let (over, off) = (Some((600.0, 300.0)), Some((20.0, 300.0)));
        let mut g = Gesture::default();
        // Resting fingers held still go where they rested.
        assert!(g.route(P::MayBegin, M::None, over, true));
        assert!(g.route(P::Changed, M::None, over, false));
        // Rested over the image, Cancelled unseen; the next gesture begins off it.
        assert!(g.route(P::MayBegin, M::None, over, true));
        assert!(!g.route(P::Began, M::None, off, false));
        assert!(!g.route(P::Changed, M::None, over, true));
        assert!(!g.route(P::Ended, M::None, over, true));
        // Rested off the image, Cancelled unseen; the next gesture begins over it.
        assert!(!g.route(P::MayBegin, M::None, off, false));
        assert!(g.route(P::Began, M::None, over, true));
        assert!(g.route(P::Changed, M::None, off, false));
        // That gesture's Ended unseen; the next begins off the image, and the next over it.
        assert!(!g.route(P::Began, M::None, off, false));
        assert!(!g.route(P::Changed, M::None, over, true));
        assert!(g.route(P::Began, M::None, over, true));
        assert!(g.route(P::Ended, M::None, off, false));
        // Where the pointer is not known, the rest decides nothing either.
        assert!(g.route(P::MayBegin, M::None, None, true));
        assert!(!g.route(P::Began, M::None, None, false));
    }

    /// An event of no phase (a notched wheel) goes where the pointer is over the image
    #[test]
    fn an_event_of_no_phase_goes_where_the_pointer_is() {
        let mut g = Gesture::default();
        assert!(g.route(
            ScrollPhase::None,
            MomentumPhase::None,
            Some((600.0, 300.0)),
            true
        ));
        assert!(!g.route(
            ScrollPhase::None,
            MomentumPhase::None,
            Some((20.0, 300.0)),
            false
        ));
        assert_eq!(g, Gesture::Idle);
    }
}
