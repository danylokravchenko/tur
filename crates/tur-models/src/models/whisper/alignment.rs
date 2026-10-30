//! Word timings from cross-attention: the steps of Transformers' `_extract_token_timestamps`
//! and the word grouping of its Whisper tokenizer.
use tur_core::Result;

/// Seconds per encoder state.
const STATE_SECONDS: f64 = 0.02;
/// Languages written without spaces, which split at Unicode boundaries alone.
const UNSPACED: [&str; 6] = ["zh", "ja", "th", "lo", "my", "yue"];
const PREPENDED: &str = "\"'“¡¿([{-";
const APPENDED: &str = "\"'.。,，!！?？:：”)]}、";
const PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// One word and when it is spoken.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub word: String,
    /// Seconds from the start of the clip.
    pub start: f64,
    pub end: f64,
}

/// A median filter over one row, reflecting at its edges; rows no longer than half the
/// filter stay as they are.
fn median_filter(row: &[f32], width: usize) -> Vec<f32> {
    let half = width / 2;
    if row.len() <= half {
        return row.to_vec();
    }
    let at = |index: isize| {
        let last = row.len().cast_signed() - 1;
        let reflected = if index < 0 {
            -index
        } else if index > last {
            2 * last - index
        } else {
            index
        };
        row[reflected.clamp(0, last).cast_unsigned()]
    };
    let mut window = vec![0.0; width];
    (0..row.len())
        .map(|center| {
            for (slot, offset) in window.iter_mut().zip(0..width) {
                *slot = at((center + offset).cast_signed() - half.cast_signed());
            }
            window.sort_by(f32::total_cmp);
            window[half]
        })
        .collect()
}

/// The path of least cost through `[tokens, states]` and, per token, the state it entered
/// at. Costs stay F32 and ties prefer staying on the token, as the reference breaks them.
fn time_warp(matrix: &[Vec<f32>]) -> Vec<usize> {
    let (rows, cols) = (matrix.len(), matrix.first().map_or(0, Vec::len));
    let mut cost = vec![f32::INFINITY; (rows + 1) * (cols + 1)];
    let mut trace = vec![2u8; (rows + 1) * (cols + 1)];
    cost[0] = 0.0;
    let at = |i: usize, j: usize| i * (cols + 1) + j;
    for j in 1..=cols {
        for i in 1..=rows {
            let (c0, c1, c2) = (
                cost[at(i - 1, j - 1)],
                cost[at(i - 1, j)],
                cost[at(i, j - 1)],
            );
            let (best, step) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            cost[at(i, j)] = -matrix[i - 1][j - 1] + best;
            trace[at(i, j)] = step;
        }
    }
    for i in 1..=rows {
        trace[at(i, 0)] = 1;
    }
    let mut entries = vec![0; rows];
    let (mut i, mut j) = (rows, cols);
    while i > 0 && j > 0 {
        entries[i - 1] = j - 1;
        match trace[at(i, j)] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    // The walk overwrites a token's state until it reaches where the token began.
    entries
}

/// Seconds into the window of each generated token, from `[selected heads][tokens][states]`
/// weights, using the first `frames / 2` states. Only the tokens fed back have weights, so
/// the last repeats the time before it.
pub(super) fn token_times(
    weights: &[Vec<Vec<f32>>],
    frames: usize,
    median_width: usize,
    generated: usize,
) -> Vec<f64> {
    let rows = weights.first().map_or(0, Vec::len);
    if rows == 0 || generated == 0 {
        return vec![0.0; generated];
    }
    let states = (frames / 2).clamp(1, weights[0][0].len());
    let mut matrix = vec![vec![0.0f32; states]; rows];
    #[allow(clippy::cast_precision_loss, reason = "counts are small")]
    let (count, heads) = (rows as f32, weights.len() as f32);
    for head in weights {
        let mut normalized = vec![vec![0.0f32; states]; rows];
        for state in 0..states {
            let mean = head.iter().map(|row| row[state]).sum::<f32>() / count;
            let variance = head
                .iter()
                .map(|row| (row[state] - mean).powi(2))
                .sum::<f32>()
                / count;
            let std = variance.sqrt();
            for (out, row) in normalized.iter_mut().zip(head) {
                out[state] = (row[state] - mean) / std;
            }
        }
        for (total, row) in matrix.iter_mut().zip(&normalized) {
            for (sum, value) in total.iter_mut().zip(median_filter(row, median_width)) {
                *sum += value / heads;
            }
        }
    }
    #[allow(clippy::cast_precision_loss, reason = "state indices are small")]
    let mut times: Vec<f64> = time_warp(&matrix)
        .into_iter()
        .map(|state| state as f64 * STATE_SECONDS)
        .collect();
    let last = times.last().copied().unwrap_or(0.0);
    times.resize(generated, last);
    times
}

/// Tokens split wherever the text decoded so far holds whole characters.
fn split_unicode(
    tokens: &[u32],
    decode: &impl Fn(&[u32]) -> Result<String>,
) -> Result<Vec<(String, Vec<usize>)>> {
    const REPLACEMENT: char = '\u{fffd}';
    let full: Vec<char> = decode(tokens)?.chars().collect();
    let mut pieces = Vec::new();
    let (mut start, mut offset) = (0, 0);
    for end in 1..=tokens.len() {
        let text = decode(&tokens[start..end])?;
        let chars: Vec<char> = text.chars().collect();
        let whole = chars
            .iter()
            .position(|&c| c == REPLACEMENT)
            .is_none_or(|at| full.get(offset + at).is_none_or(|&c| c == REPLACEMENT));
        if whole {
            offset += chars.len();
            pieces.push((text, (start..end).collect()));
            start = end;
        }
    }
    Ok(pieces)
}

/// Pieces joined until the next begins with a space or is punctuation.
fn split_spaces(pieces: Vec<(String, Vec<usize>)>) -> Vec<(String, Vec<usize>)> {
    let mut words: Vec<(String, Vec<usize>)> = Vec::new();
    for (text, indices) in pieces {
        let separate = text.starts_with(' ') || PUNCTUATION.contains(text.trim());
        match words.last_mut() {
            Some(word) if !separate => {
                word.0.push_str(&text);
                word.1.extend(indices);
            }
            _ => words.push((text, indices)),
        }
    }
    words
}

/// Attach opening punctuation to the word after it, and closing punctuation to the one before.
fn merge_punctuation(words: &mut Vec<(String, Vec<usize>)>) {
    let mut target = words.len().saturating_sub(1);
    for i in (0..target).rev() {
        if words[i].0.starts_with(' ') && PREPENDED.contains(words[i].0.trim()) {
            let (mut text, mut indices) = std::mem::take(&mut words[i]);
            text.push_str(&words[target].0);
            indices.append(&mut words[target].1);
            words[target] = (text, indices);
        } else {
            target = i;
        }
    }
    let mut target = 0;
    for j in 1..words.len() {
        if !words[target].0.ends_with(' ') && APPENDED.contains(words[j].0.as_str()) {
            let (text, mut indices) = std::mem::take(&mut words[j]);
            words[target].0.push_str(&text);
            words[target].1.append(&mut indices);
        } else {
            target = j;
        }
    }
    words.retain(|(text, _)| !text.is_empty());
}

/// Group a segment's text tokens into timed words. `times` holds the clip time of every
/// token of the segment, including its timestamps; `text` the positions of text tokens.
pub(super) fn words(
    tokens: &[u32],
    text: &[usize],
    times: &[f64],
    language: Option<&str>,
    decode: &impl Fn(&[u32]) -> Result<String>,
) -> Result<Vec<Word>> {
    let picked: Vec<u32> = text.iter().map(|&i| tokens[i]).collect();
    let pieces = split_unicode(&picked, decode)?;
    let mut grouped = if language.is_some_and(|code| UNSPACED.contains(&code)) {
        pieces
    } else {
        split_spaces(pieces)
    };
    merge_punctuation(&mut grouped);
    Ok(grouped
        .into_iter()
        .filter_map(|(word, indices)| {
            let (first, last) = (text[*indices.first()?], text[*indices.last()?]);
            Some(Word {
                word,
                start: times[first],
                end: times.get(last + 1).copied().unwrap_or(times[last]),
            })
        })
        .collect())
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    #[test]
    fn test_median_filter_reflects_at_the_edges() {
        // Sorted windows of 3 over the reflected row [5] 1 5 2 [5].
        assert_eq!(median_filter(&[1.0, 5.0, 2.0], 3), [5.0, 2.0, 5.0]);
        assert_eq!(median_filter(&[1.0, 5.0, 2.0], 7), [1.0, 5.0, 2.0]);
    }

    #[test]
    fn test_time_warp_follows_the_attention_ridge() {
        // Each token enters where the path leaves the one before; `_dynamic_time_warping`
        // gives the same entries for this matrix.
        let mut matrix = vec![vec![0.0; 12]; 3];
        matrix[0][1] = 5.0;
        matrix[1][5] = 5.0;
        matrix[2][9] = 5.0;
        assert_eq!(time_warp(&matrix), [0, 1, 5]);
    }

    /// Decodes each ID as its entry; the marked pair spells one character across tokens.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "the grouping takes a fallible tokenizer"
    )]
    fn decode(tokens: &[u32]) -> Result<String> {
        let table = [" Hello", ",", " wor", "ld", ".", " (", "ok", ")", " é"];
        let mut out = String::new();
        for (i, &token) in tokens.iter().enumerate() {
            match token {
                100 if tokens.get(i + 1) == Some(&101) => out.push('é'),
                100 | 101 if tokens.len() == 1 => out.push('\u{fffd}'),
                101 => {}
                _ => out.push_str(table[token as usize]),
            }
        }
        Ok(out)
    }

    #[test]
    fn test_tokens_group_into_words_with_their_punctuation() -> Result<()> {
        let tokens = [0, 1, 2, 3, 4, 5, 6, 7];
        let text: Vec<usize> = (0..tokens.len()).collect();
        let times: Vec<f64> = (0..=tokens.len()).map(|i| i as f64).collect();
        let grouped = words(&tokens, &text, &times, Some("en"), &decode)?;
        let summary: Vec<_> = grouped
            .iter()
            .map(|w| (w.word.as_str(), w.start, w.end))
            .collect();
        assert_eq!(
            summary,
            [
                (" Hello,", 0.0, 2.0),
                (" world.", 2.0, 5.0),
                (" (ok)", 5.0, 8.0)
            ]
        );
        Ok(())
    }

    #[test]
    fn test_characters_split_across_tokens_stay_whole() -> Result<()> {
        let times = [0.0, 1.0, 2.0];
        let grouped = words(&[100, 101], &[0, 1], &times, Some("ja"), &decode)?;
        assert_eq!(grouped.len(), 1);
        assert_eq!((grouped[0].word.as_str(), grouped[0].end), ("é", 2.0));
        Ok(())
    }
}
