//! RIFF/WAVE decoding of integer PCM and IEEE float samples into mono F32.
use tur_core::{Result, execution::invalid};

const PCM: u16 = 1;
const FLOAT: u16 = 3;
const EXTENSIBLE: u16 = 0xFFFE;

/// Decoded interleaved channels mixed to mono, and their sample rate.
pub(super) struct Decoded {
    pub(super) rate: u32,
    pub(super) samples: Vec<f32>,
}

struct Format {
    tag: u16,
    channels: usize,
    rate: u32,
    block: usize,
    bits: usize,
}

fn u16_at(bytes: &[u8], at: usize) -> Result<u16> {
    bytes
        .get(at..at + 2)
        .and_then(|b| b.try_into().ok())
        .map(u16::from_le_bytes)
        .ok_or_else(|| invalid("truncated WAV header"))
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| invalid("truncated WAV header"))
}

fn format(chunk: &[u8]) -> Result<Format> {
    let mut tag = u16_at(chunk, 0)?;
    let channels = usize::from(u16_at(chunk, 2)?);
    let rate = u32_at(chunk, 4)?;
    let block = usize::from(u16_at(chunk, 12)?);
    let bits = usize::from(u16_at(chunk, 14)?);
    if tag == EXTENSIBLE {
        // The sub-format GUID starts with the plain format tag.
        tag = u16_at(chunk, 24)?;
    }
    let supported = match tag {
        PCM => matches!(bits, 8 | 16 | 24 | 32),
        FLOAT => matches!(bits, 32 | 64),
        _ => false,
    };
    if !supported {
        return Err(invalid(format!(
            "unsupported WAV encoding (format {tag}, {bits} bits): use integer PCM or float"
        )));
    }
    if channels == 0 || rate == 0 || block != channels * bits / 8 {
        return Err(invalid("inconsistent WAV channel, rate or block layout"));
    }
    Ok(Format {
        tag,
        channels,
        rate,
        block,
        bits,
    })
}

fn sample(format: &Format, bytes: &[u8]) -> f32 {
    match (format.tag, bytes) {
        (PCM, &[b]) => (f32::from(b) - 128.0) / 128.0,
        (PCM, &[a, b]) => f32::from(i16::from_le_bytes([a, b])) / 32_768.0,
        #[allow(
            clippy::cast_precision_loss,
            reason = "24-bit samples are exact in f32"
        )]
        (PCM, &[a, b, c]) => (i32::from_le_bytes([0, a, b, c]) >> 8) as f32 / 8_388_608.0,
        #[allow(
            clippy::cast_precision_loss,
            reason = "32-bit PCM keeps f32's 24-bit precision, as every decoder does"
        )]
        (PCM, &[a, b, c, d]) => i32::from_le_bytes([a, b, c, d]) as f32 / 2f32.powi(31),
        (FLOAT, &[a, b, c, d]) => f32::from_le_bytes([a, b, c, d]),
        #[allow(
            clippy::cast_possible_truncation,
            reason = "float64 samples are narrowed to the f32 model input"
        )]
        (FLOAT, bytes @ &[_, _, _, _, _, _, _, _]) => bytes
            .try_into()
            .map_or(0.0, |b| f64::from_le_bytes(b) as f32),
        _ => 0.0,
    }
}

/// Decode a WAV file. Chunks other than `fmt ` and `data` are skipped; at most
/// `max_frames` sample frames are accepted, so a header cannot demand unbounded memory.
pub(super) fn decode(bytes: &[u8], max_frames: usize) -> Result<Decoded> {
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err(invalid("audio is not a RIFF/WAVE file"));
    }
    let mut at = 12usize;
    let mut found: Option<Format> = None;
    while at.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let id = &bytes[at..at + 4];
        let size = usize::try_from(u32_at(bytes, at + 4)?)
            .map_err(|_| invalid("WAV chunk size overflow"))?;
        let start = at + 8;
        // Streaming writers may leave the data size unset; take what is present.
        let end = start.saturating_add(size).min(bytes.len());
        let chunk = &bytes[start..end];
        match id {
            b"fmt " => found = Some(format(chunk)?),
            b"data" => {
                let format = found.ok_or_else(|| invalid("WAV data precedes its format"))?;
                return mix(&format, chunk, max_frames);
            }
            _ => {}
        }
        // Chunks are padded to an even size.
        at = end.saturating_add(size & 1);
    }
    Err(invalid("WAV file has no data chunk"))
}

fn mix(format: &Format, data: &[u8], max_frames: usize) -> Result<Decoded> {
    let frames = data.len() / format.block;
    if frames == 0 {
        return Err(invalid("WAV file holds no samples"));
    }
    if frames > max_frames {
        return Err(invalid(format!(
            "audio is {frames} frames; at most {max_frames} are accepted"
        )));
    }
    let width = format.bits / 8;
    #[allow(
        clippy::cast_precision_loss,
        reason = "channel counts are small integers"
    )]
    let scale = 1.0 / format.channels as f32;
    let samples = data
        .chunks_exact(format.block)
        .map(|frame| {
            frame
                .chunks_exact(width)
                .map(|bytes| sample(format, bytes))
                .sum::<f32>()
                * scale
        })
        .collect();
    Ok(Decoded {
        rate: format.rate,
        samples,
    })
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert decoded samples after fallible parsing"
)]
mod tests {
    use super::*;

    fn wav(tag: u16, channels: u16, bits: u16, data: &[u8]) -> Vec<u8> {
        let block = channels * bits / 8;
        let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
        out.extend_from_slice(b"LIST\x03\0\0\0abc\0");
        out.extend_from_slice(b"fmt \x10\0\0\0");
        for field in [tag, channels] {
            out.extend_from_slice(&field.to_le_bytes());
        }
        out.extend_from_slice(&8000u32.to_le_bytes());
        out.extend_from_slice(&(8000 * u32::from(block)).to_le_bytes());
        out.extend_from_slice(&block.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&u32::try_from(data.len()).unwrap_or(0).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn test_wav_formats_decode_to_mono_after_skipped_chunks() -> Result<()> {
        let pcm16 = decode(&wav(PCM, 2, 16, &[0, 0x40, 0, 0xC0, 0, 0x20, 0, 0x20]), 8)?;
        assert_eq!(pcm16.rate, 8000);
        assert_eq!(pcm16.samples, vec![0.0, 0.25]);
        let pcm24 = decode(&wav(PCM, 1, 24, &[0, 0, 0x80, 0, 0, 0x40]), 8)?;
        assert_eq!(pcm24.samples, vec![-1.0, 0.5]);
        let pcm8 = decode(&wav(PCM, 1, 8, &[0, 192]), 8)?;
        assert_eq!(pcm8.samples, vec![-1.0, 0.5]);
        let float = decode(&wav(FLOAT, 1, 32, &0.75f32.to_le_bytes()), 8)?;
        assert_eq!(float.samples, vec![0.75]);
        Ok(())
    }

    #[test]
    fn test_wav_rejects_unsupported_truncated_and_oversized_input() {
        assert!(decode(b"RIFF\0\0\0\0WAVEfmt ", 8).is_err());
        assert!(decode(&wav(2, 1, 4, &[0, 0]), 8).is_err());
        assert!(decode(&wav(PCM, 1, 16, &[0; 20]), 8).is_err());
        assert!(decode(&wav(PCM, 1, 16, &[]), 8).is_err());
        assert!(decode(b"OggS", 8).is_err());
    }
}
