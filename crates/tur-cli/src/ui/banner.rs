//! The welcome banner: a gradient that sweeps in once, in a terminal only.
use super::Console;
use ratatui::crossterm::terminal;
use std::{io::Write as _, time::Duration};

const BANNER: &str = include_str!("../../../../assets/banner.txt");
/// A 256-color ramp: Terminal.app does not render 24-bit color.
const GRADIENT: [u8; 8] = [51, 45, 39, 33, 27, 63, 99, 135];
/// The white crest leading the sweep, in columns.
const CREST: usize = 3;
const FRAME: Duration = Duration::from_millis(16);
/// Columns revealed per frame, so the whole sweep takes about half a second.
const STEP: usize = 2;

fn width() -> usize {
    BANNER
        .lines()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0)
}

/// The banner with its first `revealed` columns visible, or all of it colored at rest once
/// `revealed` passes the crest's end.
fn frame(revealed: usize) -> String {
    let columns = width().max(1);
    let mut out = String::new();
    for line in BANNER.lines() {
        out.push_str("  ");
        for (column, glyph) in line.chars().enumerate() {
            if column >= revealed {
                out.push(' ');
            } else if column + CREST >= revealed && revealed < columns + CREST {
                out.push_str("\x1b[97m");
                out.push(glyph);
            } else {
                let shade = GRADIENT[column * GRADIENT.len() / columns];
                let color = format!("\x1b[38;5;{shade}m");
                out.push_str(&color);
                out.push(glyph);
            }
        }
        out.push_str("\x1b[0m\x1b[K\n");
    }
    out
}

/// Play the banner on stderr, with `tagline` beneath it. Without color it just appears, and
/// on a terminal too narrow for it, or one nobody watches, it does not show at all.
pub(crate) fn welcome(console: &Console, tagline: &str) {
    let columns = terminal::size().map_or(0, |(columns, _)| usize::from(columns));
    if !console.interactive() || columns < width() + 2 {
        return;
    }
    let mut stderr = std::io::stderr();
    // Nothing depends on decoration reaching a closed stderr.
    if console.colored() {
        let rows = BANNER.lines().count();
        drop(write!(stderr, "\x1b[?25l"));
        let end = width() + CREST;
        for revealed in (0..end).step_by(STEP).chain([end]) {
            drop(write!(stderr, "{}\x1b[{rows}A", frame(revealed)));
            std::thread::sleep(FRAME);
        }
        drop(write!(stderr, "{}\x1b[?25h", frame(end)));
    } else {
        for line in BANNER.lines() {
            drop(writeln!(stderr, "  {line}"));
        }
    }
    console.line(&format!("  {}\n", console.paint("2", tagline)));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visible(frame: &str) -> String {
        let mut out = String::new();
        let mut escape = false;
        for glyph in frame.chars() {
            match (escape, glyph) {
                (false, '\x1b') => escape = true,
                (true, 'm' | 'K') => escape = false,
                (false, _) => out.push(glyph),
                _ => {}
            }
        }
        out
    }

    #[test]
    fn test_sweep_starts_blank_and_ends_with_the_whole_banner_at_rest() {
        assert_eq!(visible(&frame(0)).trim(), "");
        let last = frame(width() + CREST);
        let mut expected = String::new();
        for line in BANNER.lines() {
            expected.push_str("  ");
            expected.push_str(line);
            expected.push('\n');
        }
        assert_eq!(visible(&last), expected);
        assert!(!last.contains("\x1b[97m"), "the crest must leave at rest");
        assert!(frame(width() / 2).contains("\x1b[97m"));
    }
}
