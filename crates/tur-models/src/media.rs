//! Non-text prompt inputs, as chat messages carry them and models place them.
use crate::{audio::Audio, image::Image};

/// One clip or image of a request.
#[derive(Clone, Copy, Debug)]
pub enum Media<'a> {
    Audio(&'a Audio),
    Image(&'a Image),
}

impl Media<'_> {
    /// Identifies the content, as prefix reuse requires, whatever file it came from.
    pub fn fingerprint(&self) -> u64 {
        match self {
            Self::Audio(clip) => clip.fingerprint(),
            Self::Image(image) => image.fingerprint(),
        }
    }
}

/// How a model frames one input in its prompt. The chat template renders `placeholder` once
/// per input; the input then occupies `begin`, one position per soft token and `end`, the
/// framing being whatever the template did not write itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaPrompt {
    pub placeholder: u32,
    pub begin: Option<u32>,
    pub end: Option<u32>,
}

/// The soft tokens of one input in a prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaLayout {
    pub tokens: usize,
    /// `(rows, columns)` of soft tokens when text rotary reads them as a grid.
    pub grid: Option<(usize, usize)>,
}
