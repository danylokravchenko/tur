//! HEVC in MP4: finding a track's decoder configuration, and pictures as RGB.
//!
//! The MP4 reader skips `hvc1` sample entries, the kind phones write, and discards the `hvcC`
//! box of `hev1` ones, so this walks to them itself, with every length checked against its parent.
use super::{Frame, bad_file};
use tur_core::{Result, TurError};

/// What a track's `hvcC` box holds that decoding needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Config {
    /// Bytes of the length prefix on each NAL unit of a sample.
    pub(super) length_size: u8,
    /// VPS, SPS and PPS with Annex B start codes.
    pub(super) parameters: Vec<u8>,
}

pub(super) fn error(error: &rust_h265::DecodeError) -> TurError {
    bad_file(format!("{error:?}"))
}

/// The boxes directly inside `data`, as (type, payload). Stops at the first malformed one: the
/// caller then simply doesn't find what it looks for.
fn boxes(mut data: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    std::iter::from_fn(move || {
        let (size, rest) = data.split_first_chunk::<4>()?;
        let (kind, rest) = rest.split_first_chunk::<4>()?;
        let (payload, next) = match u32::from_be_bytes(*size) {
            0 => (rest, &[][..]),
            1 => {
                let (large, rest) = rest.split_first_chunk::<8>()?;
                let length = usize::try_from(u64::from_be_bytes(*large)).ok()?;
                rest.split_at_checked(length.checked_sub(16)?)?
            }
            size => rest.split_at_checked(usize::try_from(size).ok()?.checked_sub(8)?)?,
        };
        data = next;
        Some((kind.as_slice(), payload))
    })
}

fn child<'a>(data: &'a [u8], kind: &[u8]) -> Option<&'a [u8]> {
    boxes(data)
        .find(|(found, _)| *found == kind)
        .map(|(_, payload)| payload)
}

/// A `tkhd` payload's track ID, at an offset that depends on its version.
fn track_id(tkhd: &[u8]) -> Option<u32> {
    let at = if *tkhd.first()? == 1 { 20 } else { 12 };
    Some(u32::from_be_bytes(*tkhd.get(at..)?.first_chunk::<4>()?))
}

impl Config {
    /// The configuration of track `id` when it is HEVC, with its stored `(width, height)`.
    pub(super) fn find(file: &[u8], id: u32) -> Result<Option<(Self, (usize, usize))>> {
        let trak = child(file, b"moov")
            .into_iter()
            .flat_map(boxes)
            .find(|(kind, trak)| {
                *kind == b"trak" && child(trak, b"tkhd").and_then(track_id) == Some(id)
            });
        let entry = trak.and_then(|(_, trak)| {
            let mdia = child(trak, b"mdia")?;
            let stbl = child(child(mdia, b"minf")?, b"stbl")?;
            // A full box's four bytes and the entry count, then the sample entries.
            boxes(child(stbl, b"stsd")?.get(8..)?)
                .find(|(kind, _)| *kind == b"hvc1" || *kind == b"hev1")
        });
        let Some((_, entry)) = entry else {
            return Ok(None);
        };
        let field = |at: usize| -> Result<usize> {
            let bytes = entry
                .get(at..)
                .and_then(<[u8]>::first_chunk::<2>)
                .ok_or_else(|| bad_file("truncated HEVC sample entry"))?;
            Ok(usize::from(u16::from_be_bytes(*bytes)))
        };
        let stored = (field(24)?, field(26)?);
        // A visual sample entry's fixed fields take 78 bytes.
        let record = entry
            .get(78..)
            .and_then(|children| child(children, b"hvcC"))
            .ok_or_else(|| bad_file("HEVC sample entry without a configuration"))?;
        Ok(Some((Self::parse(record)?, stored)))
    }

    /// An `HEVCDecoderConfigurationRecord`.
    fn parse(record: &[u8]) -> Result<Self> {
        let truncated = || bad_file("truncated HEVC configuration");
        let length_size = (record.get(21).ok_or_else(truncated)? & 3) + 1;
        let arrays = *record.get(22).ok_or_else(truncated)?;
        let mut rest = record.get(23..).ok_or_else(truncated)?;
        let mut parameters = Vec::new();
        for _ in 0..arrays {
            let (head, tail) = rest.split_first_chunk::<3>().ok_or_else(truncated)?;
            rest = tail;
            for _ in 0..u16::from_be_bytes([head[1], head[2]]) {
                let (size, tail) = rest.split_first_chunk::<2>().ok_or_else(truncated)?;
                let (unit, tail) = tail
                    .split_at_checked(usize::from(u16::from_be_bytes(*size)))
                    .ok_or_else(truncated)?;
                parameters.extend_from_slice(&[0, 0, 0, 1]);
                parameters.extend_from_slice(unit);
                rest = tail;
            }
        }
        if parameters.is_empty() {
            return Err(bad_file("HEVC configuration without parameter sets"));
        }
        Ok(Self {
            length_size,
            parameters,
        })
    }
}

/// A decoded picture on its way to RGB.
pub(super) struct Picture<'a>(pub(super) &'a rust_h265::Frame);

/// A plane's samples at 8 bits. Phones' HDR video is 10-bit; the models read 8.
fn plane(data: &rust_h265::PixelData, depth: u8) -> Result<std::borrow::Cow<'_, [u8]>> {
    if let Some(samples) = data.as_u8() {
        return Ok(samples.into());
    }
    let samples = data
        .as_u16()
        .ok_or_else(|| bad_file("a picture without samples"))?;
    let shift = u32::from(depth.saturating_sub(8));
    #[allow(clippy::cast_possible_truncation, reason = "shifted down to 8 bits")]
    Ok(samples
        .iter()
        .map(|&v| (v >> shift) as u8)
        .collect::<Vec<u8>>()
        .into())
}

impl Frame for Picture<'_> {
    fn size(&self) -> (usize, usize) {
        (self.0.width as usize, self.0.height as usize)
    }

    /// Limited-range BT.601, the conversion `PyAV` applies to a stream by default and
    /// `OpenH264`'s, so a clip reads alike in either codec. Chroma is repeated per 2×2 block.
    fn rgb(&self) -> Result<Vec<u8>> {
        if self.0.chroma_format_idc != 1 {
            return Err(bad_file("only 4:2:0 HEVC is supported"));
        }
        let (width, height) = self.size();
        let depth = self.0.bit_depth;
        let (y, u, v) = (
            plane(&self.0.y, depth)?,
            plane(&self.0.u, depth)?,
            plane(&self.0.v, depth)?,
        );
        let chroma = width.div_ceil(2);
        if y.len() < width * height || u.len() < chroma * height.div_ceil(2) || v.len() < u.len() {
            return Err(bad_file("a picture smaller than its size"));
        }
        let mut rgb = Vec::with_capacity(width * height * 3);
        for row in 0..height {
            for column in 0..width {
                let at = row / 2 * chroma + column / 2;
                let luma = (i32::from(y[row * width + column]) - 16) * 298 + 128;
                let (cb, cr) = (i32::from(u[at]) - 128, i32::from(v[at]) - 128);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped to 0..=255"
                )]
                let settle = |value: i32| (value >> 8).clamp(0, 255) as u8;
                rgb.extend_from_slice(&[
                    settle(luma + 409 * cr),
                    settle(luma - 100 * cb - 208 * cr),
                    settle(luma + 516 * cb),
                ]);
            }
        }
        Ok(rgb)
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn sized(kind: &[u8], payload: &[u8]) -> Vec<u8> {
        let size = u32::try_from(payload.len() + 8).unwrap_or(u32::MAX);
        [&size.to_be_bytes()[..], kind, payload].concat()
    }

    #[test]
    fn test_boxes_stop_at_lengths_beyond_their_parent() {
        let good = sized(b"free", &[1, 2]);
        let mut data = good;
        data.extend_from_slice(&[0, 0, 1, 0, b'b', b'a', b'd', b'!', 9]);
        let found: Vec<_> = boxes(&data).collect();
        assert_eq!(found, [(&b"free"[..], &[1u8, 2][..])]);
        // A size smaller than its own header is malformed too.
        assert_eq!(boxes(&[0, 0, 0, 4, b'a', b'b', b'c', b'd']).count(), 0);
    }

    #[test]
    fn test_configuration_records_are_read_and_truncations_rejected() -> Result<()> {
        let mut record = vec![0u8; 21];
        record.push(0xFF); // Length size 4.
        record.push(1); // One array of one unit.
        record.extend_from_slice(&[32, 0, 1, 0, 2, 0x40, 0x01]);
        let config = Config::parse(&record)?;
        assert_eq!(config.length_size, 4);
        assert_eq!(config.parameters, [0, 0, 0, 1, 0x40, 0x01]);
        for end in 0..record.len() {
            assert!(Config::parse(&record[..end]).is_err(), "{end}");
        }
        Ok(())
    }
}
