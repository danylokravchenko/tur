//! Image input: decoding to RGB and resizing as the models' image processors do.
mod resize;

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};
use std::{io::Cursor, sync::Arc};
use tur_core::{Result, execution::invalid};

/// Longest side of a decoded image.
pub const MAX_SIDE: u32 = 16_384;
/// Host bytes one decode may allocate: an 8K-square RGBA image fits. Decoders check this
/// before allocating, so a small file can't claim a huge bitmap.
const MAX_DECODED_BYTES: u64 = 1 << 28;

/// One RGB image, shared cheaply between a request and its model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    width: u32,
    height: u32,
    /// `height` rows of `width` RGB byte triples.
    pixels: Arc<[u8]>,
}

fn decode_error(error: impl std::fmt::Display) -> tur_core::TurError {
    invalid(format!("undecodable image: {error}"))
}

impl Image {
    /// Decode a PNG, JPEG, WebP or GIF (first frame) file. EXIF orientation is applied, and
    /// an alpha channel dropped, as Transformers' `do_convert_rgb` drops it.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(decode_error)?;
        if !matches!(
            reader.format(),
            Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif)
        ) {
            return Err(invalid(
                "unsupported image format: send PNG, JPEG, WebP or GIF",
            ));
        }
        let mut limits = Limits::default();
        limits.max_image_width = Some(MAX_SIDE);
        limits.max_image_height = Some(MAX_SIDE);
        limits.max_alloc = Some(MAX_DECODED_BYTES);
        reader.limits(limits);
        let mut decoder = reader.into_decoder().map_err(decode_error)?;
        let orientation = decoder.orientation().map_err(decode_error)?;
        let mut image = DynamicImage::from_decoder(decoder).map_err(decode_error)?;
        image.apply_orientation(orientation);
        let rgb = image.into_rgb8();
        let (width, height) = rgb.dimensions();
        Self::from_rgb(width, height, rgb.into_raw())
    }

    /// An image of `height` rows of `width` RGB byte triples.
    pub fn from_rgb(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| invalid("image size overflow"))?;
        if width == 0 || height == 0 || pixels.len() != expected {
            return Err(invalid("image size does not match its pixels"));
        }
        Ok(Self {
            width,
            height,
            pixels: pixels.into(),
        })
    }

    pub fn width(&self) -> usize {
        self.width as usize
    }

    pub fn height(&self) -> usize {
        self.height as usize
    }

    /// `height` rows of `width` RGB byte triples.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// 64-bit FNV-1a of the size and pixels: images are told apart by content, as prefix
    /// reuse requires, not by how they were encoded.
    pub fn fingerprint(&self) -> u64 {
        let size = (u64::from(self.width) << 32) | u64::from(self.height);
        size.to_le_bytes()
            .iter()
            .chain(self.pixels.iter())
            .fold(0xcbf2_9ce4_8422_2325u64, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
            })
    }

    /// Interleaved RGB of the image at `height` × `width`, resampled as the models' image
    /// processors do. The image's own size returns its pixels.
    pub(crate) fn resized(
        &self,
        height: usize,
        width: usize,
    ) -> Result<std::borrow::Cow<'_, [u8]>> {
        if height == 0 || width == 0 {
            return Err(invalid("image resized to nothing"));
        }
        if (height, width) == (self.height(), self.width()) {
            return Ok(std::borrow::Cow::Borrowed(&self.pixels));
        }
        Ok(std::borrow::Cow::Owned(resize::bicubic(
            &self.pixels,
            (self.height(), self.width()),
            (height, width),
        )))
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert decoded pixels after fallible setup"
)]
mod tests {
    use super::*;
    use image::{ImageEncoder, codecs::png::PngEncoder};

    fn png(
        width: u32,
        height: u32,
        color: image::ExtendedColorType,
        data: &[u8],
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        PngEncoder::new(&mut out)
            .write_image(data, width, height, color)
            .map_err(decode_error)?;
        Ok(out)
    }

    #[test]
    fn test_decoding_keeps_rgb_and_drops_alpha() -> Result<()> {
        let rgba = [10, 20, 30, 0, 40, 50, 60, 255];
        let image = Image::decode(&png(2, 1, image::ExtendedColorType::Rgba8, &rgba)?)?;
        assert_eq!((image.width(), image.height()), (2, 1));
        assert_eq!(&*image.pixels, &[10, 20, 30, 40, 50, 60]);
        Ok(())
    }

    #[test]
    fn test_decoding_rejects_other_formats_and_claimed_sizes_beyond_the_limit() {
        assert!(Image::decode(b"RIFF....WAVEfmt ").is_err());
        assert!(Image::decode(b"not an image").is_err());
        // A valid header that claims more than the decode limit, with no pixels behind it.
        let mut header = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13];
        header.extend_from_slice(b"IHDR");
        header.extend_from_slice(&30_000u32.to_be_bytes());
        header.extend_from_slice(&30_000u32.to_be_bytes());
        header.extend_from_slice(&[8, 2, 0, 0, 0, 0, 0, 0, 0]);
        assert!(Image::decode(&header).is_err());
    }

    #[test]
    fn test_fingerprints_differ_by_size_and_content() -> Result<()> {
        let wide = Image::from_rgb(2, 1, vec![1; 6])?;
        let tall = Image::from_rgb(1, 2, vec![1; 6])?;
        let other = Image::from_rgb(2, 1, vec![2; 6])?;
        assert_ne!(wide.fingerprint(), tall.fingerprint());
        assert_ne!(wide.fingerprint(), other.fingerprint());
        assert_eq!(
            wide.fingerprint(),
            Image::from_rgb(2, 1, vec![1; 6])?.fingerprint()
        );
        Ok(())
    }
}
