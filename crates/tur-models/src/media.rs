//! Non-text prompt inputs, as chat messages carry them and models place them.
use crate::{audio::Audio, image::Image, video::Video};
use sha2::{Digest as _, Sha256};

/// A cryptographic hash of an input's content. Prefix reuse and encoded rows are shared by
/// content, so a client must not be able to craft a collision with someone else's input.
pub type Digest = [u8; 32];

/// The digest of `parts` in order. The first part names the kind of input.
pub(crate) fn digest(parts: &[&[u8]]) -> Digest {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// One clip, image or video of a request.
#[derive(Clone, Copy, Debug)]
pub enum Media<'a> {
    Audio(&'a Audio),
    Image(&'a Image),
    Video(&'a Video),
}

impl Media<'_> {
    /// Identifies the content, whatever file it came from.
    pub fn digest(&self) -> Digest {
        match self {
            Self::Audio(clip) => clip.digest(),
            Self::Image(image) => image.digest(),
            Self::Video(video) => video.digest(),
        }
    }
}

/// How a model frames one input in its prompt. The chat template renders `placeholder` once
/// per input; each of the input's segments then occupies its label, `begin`, one position per
/// soft token and `end`, the framing being whatever the template did not write itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaPrompt {
    pub placeholder: u32,
    pub begin: Option<u32>,
    pub end: Option<u32>,
}

/// One run of an input's soft tokens in a prompt: a video has one per sampled step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaSegment {
    /// Text the prompt holds before the segment, such as a video frame's timestamp.
    pub label: String,
    pub tokens: usize,
    /// `(rows, columns)` of soft tokens when text rotary reads them as a grid.
    pub grid: Option<(usize, usize)>,
}

/// The soft tokens of one input in a prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaLayout {
    pub segments: Vec<MediaSegment>,
}

impl MediaLayout {
    /// One unlabelled run.
    pub fn single(tokens: usize, grid: Option<(usize, usize)>) -> Self {
        Self {
            segments: vec![MediaSegment {
                label: String::new(),
                tokens,
                grid,
            }],
        }
    }

    /// Soft tokens of every segment.
    pub fn tokens(&self) -> usize {
        self.segments.iter().map(|segment| segment.tokens).sum()
    }
}
