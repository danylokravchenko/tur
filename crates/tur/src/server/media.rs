//! Request media: base64 payloads, data URLs and `multipart/form-data` bodies.
use crate::{Result, TurError};

fn bad(message: impl Into<String>) -> TurError {
    tur_core::execution::invalid(message)
}

/// Decode standard or URL-safe base64, with or without padding; whitespace is ignored.
pub fn base64(text: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut buffer, mut bits) = (0u32, 0u32);
    let mut padding = false;
    for byte in text.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => {
                padding = true;
                continue;
            }
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return Err(bad("invalid base64 payload")),
        };
        if padding {
            return Err(bad("base64 data after padding"));
        }
        buffer = (buffer << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            #[allow(clippy::cast_possible_truncation, reason = "keeps the completed byte")]
            out.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }
    // A lone trailing sextet cannot complete a byte.
    if bits >= 6 {
        return Err(bad("truncated base64 payload"));
    }
    Ok(out)
}

/// The bytes of a `data:<type>;base64,<payload>` URL. Remote URLs are rejected: the
/// server never fetches media on a client's behalf.
pub fn data_url(url: &str) -> Result<(String, Vec<u8>)> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| bad("only data: URLs are accepted for media"))?;
    let (header, payload) = rest
        .split_once(',')
        .ok_or_else(|| bad("malformed data URL"))?;
    let media = header
        .strip_suffix(";base64")
        .ok_or_else(|| bad("data URLs must be base64-encoded"))?;
    Ok((media.to_ascii_lowercase(), base64(payload)?))
}

/// One part of a `multipart/form-data` body.
pub struct Field<'a> {
    pub name: String,
    pub filename: Option<String>,
    pub content_type: Option<String>,
    pub data: &'a [u8],
}

fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then_some(value.trim())
    })
}

fn parameter(value: &str, name: &str) -> Option<String> {
    value.split(';').skip(1).find_map(|part| {
        let (key, raw) = part.split_once('=')?;
        (key.trim() == name).then(|| raw.trim().trim_matches('"').to_owned())
    })
}

/// The multipart boundary of a `Content-Type` header.
pub fn boundary(content_type: &str) -> Result<String> {
    if !content_type
        .trim()
        .to_ascii_lowercase()
        .starts_with("multipart/form-data")
    {
        return Err(bad("expected a multipart/form-data body"));
    }
    parameter(content_type, "boundary")
        .filter(|b| !b.is_empty() && b.len() <= 70)
        .ok_or_else(|| bad("multipart body without a boundary"))
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    haystack
        .get(from..)?
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|at| at + from)
}

/// Split a `multipart/form-data` body into its fields (RFC 7578), borrowing their data.
pub fn multipart<'a>(body: &'a [u8], boundary: &str) -> Result<Vec<Field<'a>>> {
    let delimiter = format!("--{boundary}");
    let separator = format!("\r\n--{boundary}");
    let mut at = find(body, delimiter.as_bytes(), 0)
        .ok_or_else(|| bad("multipart body has no parts"))?
        + delimiter.len();
    let mut fields = Vec::new();
    loop {
        if body.get(at..at + 2) == Some(b"--") {
            return Ok(fields);
        }
        if body.get(at..at + 2) != Some(b"\r\n") {
            return Err(bad("malformed multipart delimiter"));
        }
        let start = at + 2;
        let end_headers =
            find(body, b"\r\n\r\n", start).ok_or_else(|| bad("multipart part without headers"))?;
        let headers = std::str::from_utf8(&body[start..end_headers])
            .map_err(|_| bad("multipart headers must be UTF-8"))?;
        let data_start = end_headers + 4;
        let data_end = find(body, separator.as_bytes(), data_start)
            .ok_or_else(|| bad("unterminated multipart part"))?;
        let disposition = header_value(headers, "content-disposition")
            .ok_or_else(|| bad("multipart part without a content disposition"))?;
        fields.push(Field {
            name: parameter(disposition, "name")
                .ok_or_else(|| bad("multipart part without a name"))?,
            filename: parameter(disposition, "filename"),
            content_type: header_value(headers, "content-type").map(str::to_owned),
            data: &body[data_start..data_end],
        });
        at = data_end + separator.len();
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert decoded payloads after fallible parsing"
)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_and_data_urls_decode_and_reject_garbage() -> Result<()> {
        assert_eq!(base64("aGVsbG8=")?, b"hello");
        assert_eq!(base64("aGVsbG8")?, b"hello");
        assert_eq!(base64("aGVs\nbG8h")?, b"hello!");
        assert_eq!(base64("-_8=")?, vec![0xfb, 0xff]);
        assert!(base64("aGVsbG8*").is_err());
        assert!(base64("a").is_err());
        assert!(base64("aA==aA").is_err());
        let (media, bytes) = data_url("data:audio/WAV;base64,UklGRg==")?;
        assert_eq!(
            (media.as_str(), bytes.as_slice()),
            ("audio/wav", &b"RIFF"[..])
        );
        assert!(data_url("https://example.com/a.wav").is_err());
        assert!(data_url("data:audio/wav,RIFF").is_err());
        Ok(())
    }

    #[test]
    fn test_multipart_fields_keep_binary_data_and_metadata() -> Result<()> {
        let body = b"--xyz\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper\r\n\
--xyz\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
Content-Type: audio/wav\r\n\r\nRI\r\nFF\r\n--xyz--\r\n";
        let delimiter = boundary("multipart/form-data; boundary=xyz")?;
        let fields = multipart(body, &delimiter)?;
        assert_eq!(fields.len(), 2);
        assert_eq!(
            (fields[0].name.as_str(), fields[0].data),
            ("model", &b"whisper"[..])
        );
        assert_eq!(fields[1].filename.as_deref(), Some("a.wav"));
        assert_eq!(fields[1].content_type.as_deref(), Some("audio/wav"));
        assert_eq!(fields[1].data, b"RI\r\nFF");
        assert!(boundary("application/json").is_err());
        assert!(multipart(b"--xyz\r\nno headers", "xyz").is_err());
        Ok(())
    }
}
