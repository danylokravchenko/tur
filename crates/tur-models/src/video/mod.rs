//! Video input: H.264 in MP4, decoded only as far as the frames a model samples need.
use crate::media::{Digest, digest};
use mp4::{Mp4Reader, TrackType};
use openh264::{
    OpenH264API,
    decoder::{Decoder, DecoderConfig, Flush},
    formats::YUVSource,
};
use std::{io::Cursor, panic::AssertUnwindSafe, sync::Arc};
use tur_core::{Result, execution::invalid};

/// Longest side of a video frame.
pub const MAX_SIDE: usize = 4096;
/// Most frames a video may hold: an hour at 30 fps.
const MAX_FRAMES: usize = 108_000;

/// Where a video's frames come from.
#[derive(Clone, Debug, PartialEq)]
enum Source {
    Mp4(Arc<[u8]>),
    /// RGB frames, as a caller that decoded them passes them.
    Frames(Arc<[Vec<u8>]>),
}

/// One video, shared cheaply between a request and its model. Holds its file, not its frames:
/// which frames to decode depends on the model.
#[derive(Clone, Debug, PartialEq)]
pub struct Video {
    source: Source,
    width: usize,
    height: usize,
    frames: usize,
    fps: f64,
    digest: Digest,
}

fn bad_file(error: impl std::fmt::Display) -> tur_core::TurError {
    invalid(format!("undecodable video: {error}"))
}

/// Run a parser or decoder over untrusted bytes, turning its panic into a request's error so
/// that a bad file can't take down the thread encoding other requests' media.
fn guarded<T>(run: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(AssertUnwindSafe(run))
        .unwrap_or_else(|_| Err(bad_file("the file is malformed")))
}

fn open(bytes: &[u8]) -> Result<(Mp4Reader<Cursor<&[u8]>>, u32)> {
    let size = bytes.len() as u64;
    let reader = Mp4Reader::read_header(Cursor::new(bytes), size).map_err(bad_file)?;
    if reader.is_fragmented() {
        return Err(invalid("fragmented MP4 files are not supported"));
    }
    let track = reader
        .tracks()
        .values()
        .filter(|track| matches!(track.track_type(), Ok(TrackType::Video)))
        .map(mp4::Mp4Track::track_id)
        .min()
        .ok_or_else(|| invalid("the file has no video track"))?;
    Ok((reader, track))
}

impl Video {
    /// Read an MP4 file's H.264 video track. Its frames are decoded when a model needs them.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        guarded(|| {
            let (reader, id) = open(bytes)?;
            let track = reader
                .tracks()
                .get(&id)
                .ok_or_else(|| invalid("the file has no video track"))?;
            if track.box_type().map_err(bad_file)? != mp4::BoxType::Avc1Box.into() {
                return Err(invalid(
                    "unsupported video codec: send H.264 in an MP4 file",
                ));
            }
            let frames = track.sample_count() as usize;
            // The commonest frame duration: an edit list or a long last frame skews the track's
            // total, and timestamps should read as the file's nominal rate.
            let delta = track
                .trak
                .mdia
                .minf
                .stbl
                .stts
                .entries
                .iter()
                .max_by_key(|entry| entry.sample_count)
                .map_or(0, |entry| entry.sample_delta);
            if frames < 2 || delta == 0 {
                return Err(invalid("a video needs at least two frames and a rate"));
            }
            Self::checked(
                Source::Mp4(bytes.into()),
                (usize::from(track.width()), usize::from(track.height())),
                frames,
                f64::from(track.timescale()) / f64::from(delta),
                digest(&[b"video", bytes]),
            )
        })
    }

    /// A video of RGB `frames`, each `height` rows of `width` byte triples.
    pub fn from_frames(
        width: usize,
        height: usize,
        fps: f64,
        frames: Vec<Vec<u8>>,
    ) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| invalid("video size overflow"))?;
        if frames.iter().any(|frame| frame.len() != expected) {
            return Err(invalid("video size does not match its frames"));
        }
        if frames.len() < 2 {
            return Err(invalid("a video needs at least two frames"));
        }
        let size = [width as u64, height as u64, fps.to_bits()].map(u64::to_le_bytes);
        let mut parts: Vec<&[u8]> = vec![b"frames", &size[0], &size[1], &size[2]];
        parts.extend(frames.iter().map(Vec::as_slice));
        let digest = digest(&parts);
        let count = frames.len();
        Self::checked(
            Source::Frames(frames.into()),
            (width, height),
            count,
            fps,
            digest,
        )
    }

    fn checked(
        source: Source,
        (width, height): (usize, usize),
        frames: usize,
        fps: f64,
        digest: Digest,
    ) -> Result<Self> {
        if width == 0 || height == 0 || width.max(height) > MAX_SIDE {
            return Err(invalid(format!(
                "video frames must fit in {MAX_SIDE}×{MAX_SIDE} pixels; these are {width}×{height}"
            )));
        }
        if frames > MAX_FRAMES || !fps.is_finite() || fps <= 0.0 {
            return Err(invalid("a video's frame count or rate is out of range"));
        }
        Ok(Self {
            source,
            width,
            height,
            frames,
            fps,
            digest,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Frames the video holds.
    pub fn frame_count(&self) -> usize {
        self.frames
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Identifies the file or frames, as prefix reuse requires.
    pub fn digest(&self) -> Digest {
        self.digest
    }

    /// RGB frames at nondecreasing `indices`, in that order.
    pub fn frames(&self, indices: &[usize]) -> Result<Vec<Vec<u8>>> {
        if indices.windows(2).any(|pair| pair[0] > pair[1])
            || indices.last().is_some_and(|&last| last >= self.frames)
        {
            return Err(invalid("video frame indices out of order or range"));
        }
        match &self.source {
            Source::Frames(frames) => Ok(indices.iter().map(|&at| frames[at].clone()).collect()),
            Source::Mp4(bytes) => guarded(|| decode_frames(bytes, indices, self.frames)),
        }
    }
}

/// An access unit's length-prefixed NAL units with Annex B start codes, which the decoder reads.
fn annex_b(sample: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let mut rest = sample;
    while !rest.is_empty() {
        let (head, tail) = rest
            .split_first_chunk::<4>()
            .ok_or_else(|| bad_file("truncated NAL unit length"))?;
        let length = u32::from_be_bytes(*head) as usize;
        let unit = tail
            .get(..length)
            .ok_or_else(|| bad_file("NAL unit beyond its sample"))?;
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(unit);
        rest = &tail[length..];
    }
    Ok(())
}

/// Collects decoded frames in presentation order, keeping the ones asked for.
struct Collector<'a> {
    indices: &'a [usize],
    wanted: usize,
    next: usize,
    out: Vec<Vec<u8>>,
}

impl Collector<'_> {
    fn done(&self) -> bool {
        self.wanted >= self.indices.len()
    }

    fn take(&mut self, frame: &impl Frame) {
        while self.indices.get(self.wanted) == Some(&self.next) {
            self.out.push(frame.rgb());
            self.wanted += 1;
        }
        self.next += 1;
    }
}

trait Frame {
    fn rgb(&self) -> Vec<u8>;
}

impl Frame for openh264::decoder::DecodedYUV<'_> {
    fn rgb(&self) -> Vec<u8> {
        let (width, height) = self.dimensions();
        let mut rgb = vec![0u8; width * height * 3];
        self.write_rgb8(&mut rgb);
        rgb
    }
}

fn decode_frames(bytes: &[u8], indices: &[usize], frames: usize) -> Result<Vec<Vec<u8>>> {
    let (mut reader, id) = open(bytes)?;
    let track = reader
        .tracks()
        .get(&id)
        .ok_or_else(|| bad_file("no video track"))?;
    // Flushing after each packet, the crate's default, emits frames out of display order and
    // fails on the first B-frames; the decoder reorders them itself when left to fill up.
    let mut decoder = Decoder::with_api_config(
        OpenH264API::from_source(),
        DecoderConfig::new().flush_after_decode(Flush::NoFlush),
    )
    .map_err(bad_file)?;
    let mut packet = Vec::new();
    for parameters in [
        track.sequence_parameter_set().map_err(bad_file)?,
        track.picture_parameter_set().map_err(bad_file)?,
    ] {
        packet.extend_from_slice(&[0, 0, 0, 1]);
        packet.extend_from_slice(parameters);
    }
    let mut collector = Collector {
        indices,
        wanted: 0,
        next: 0,
        out: Vec::with_capacity(indices.len()),
    };
    for sample in 1..=u32::try_from(frames).map_err(|_| bad_file("too many frames"))? {
        if collector.done() {
            return Ok(collector.out);
        }
        let sample = reader
            .read_sample(id, sample)
            .map_err(bad_file)?
            .ok_or_else(|| bad_file("missing sample"))?;
        annex_b(&sample.bytes, &mut packet)?;
        if let Some(frame) = decoder.decode(&packet).map_err(bad_file)? {
            collector.take(&frame);
        }
        packet.clear();
    }
    for frame in decoder.flush_remaining().map_err(bad_file)? {
        collector.take(&frame);
    }
    if !collector.done() {
        return Err(bad_file("fewer decoded frames than samples"));
    }
    Ok(collector.out)
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    #[test]
    fn test_garbage_and_truncated_files_are_rejected_as_errors() {
        assert!(Video::decode(b"not a video at all").is_err());
        assert!(Video::decode(&[0u8; 4096]).is_err());
    }

    #[test]
    fn test_annex_b_rejects_units_beyond_their_sample() -> Result<()> {
        let mut out = Vec::new();
        annex_b(&[0, 0, 0, 2, 7, 8, 0, 0, 0, 1, 9], &mut out)?;
        assert_eq!(out, [0, 0, 0, 1, 7, 8, 0, 0, 0, 1, 9]);
        assert!(annex_b(&[0, 0, 0, 9, 7], &mut out).is_err());
        assert!(annex_b(&[0, 0], &mut out).is_err());
        Ok(())
    }

    #[test]
    fn test_frames_come_back_at_their_indices_and_bad_indices_fail() -> Result<()> {
        let frames: Vec<Vec<u8>> = (0..4u8).map(|v| vec![v; 3]).collect();
        let video = Video::from_frames(1, 1, 2.0, frames)?;
        assert_eq!(video.frames(&[1, 3])?, vec![vec![1; 3], vec![3; 3]]);
        assert!(video.frames(&[3, 1]).is_err());
        assert!(video.frames(&[4]).is_err());
        assert!(Video::from_frames(1, 1, 2.0, vec![vec![0; 3]]).is_err());
        Ok(())
    }
}
